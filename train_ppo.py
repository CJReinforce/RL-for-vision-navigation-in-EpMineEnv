import datetime
import os
import  random
from time import time

import numpy as np
from tianshou.env.venvs import BaseVectorEnv
import torch
from envs.SingleAgent.mine_toy import EpMineEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.discrete import Critic
from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ).to(device)
        self.output_dim = 256

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        shape = obs.shape
        obs = obs.reshape(-1, *shape[-3:])
        obs = obs.permute(0, 3, 1, 2).to(self.device)
        out = self.net(obs/255)  # obs normalization
        out = out.reshape(shape[0], -1)
        return out


class PreProcessNet(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.image_feature = CNN(device).to(device)
        self.net_image = nn.Sequential(
            self.image_feature,
            nn.Linear(self.image_feature.output_dim, 64),
            nn.ReLU(),
        ).to(device)
        self.net_state = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
        ).to(device)
        self.output_dim = 128
    
    def forward(self, obs, state=None, info={}):
        img = torch.as_tensor(obs['image'], dtype=torch.float32, device=self.device)
        obs_state = torch.as_tensor(obs['state'], dtype=torch.float32, device=self.device)
        img_feature = self.net_image(img)
        state_feature = self.net_state(obs_state)
        out = torch.cat([img_feature, state_feature], axis=1).to(self.device)
        return out, state


class VectorMineEnvNormObs(VectorEnvNormObs):
    def __init__(self, venv: BaseVectorEnv, update_obs_rms: bool = True) -> None:
        super().__init__(venv, update_obs_rms)

    def reset(self, id=None, **kwargs):
        obs, info = self.venv.reset(id, **kwargs)
        obs_state = np.array([i['state'] for i in obs], dtype=np.float32)

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_state)
        obs_state = self._norm_obs(obs_state)
        obs = [dict(image=i['image'], state=j) for i, j in zip(obs, obs_state)]
        return obs, info

    def step(self, action: np.ndarray, id=None):
        step_results = self.venv.step(action, id)
        next_state = step_results[0]
        obs_state = np.array([i['state'] for i in next_state], dtype=np.float32)

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_state)
        
        obs_state = self._norm_obs(obs_state)
        next_state = [dict(image=i['image'], state=j) for i, j in zip(next_state, obs_state)]
        return (next_state, *step_results[1:])
    

def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(seed: int, training_num: int, test_num: int, obs_norm: bool = True):
    file_name = '/root/data/cj/program/EpMineEnv/envs/SingleAgent/MineField_Linux-0510-random/drl.x86_64'
    env = EpMineEnv(file_name=file_name, port=2000,
                    no_graph=True, only_state=False, only_image=False)
    train_envs = ShmemVectorEnv(
        [lambda: EpMineEnv(file_name=file_name, port=3000,
                           no_graph=True, only_state=False, only_image=False) for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv(
        [lambda: EpMineEnv(file_name=file_name, port=4000,
                           no_graph=True, only_state=False, only_image=False) for _ in range(test_num)]
    )
    
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    if obs_norm:
        train_envs = VectorMineEnvNormObs(train_envs)
        test_envs = VectorMineEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    
    return env, train_envs, test_envs


if __name__ == "__main__":
    seed = 42
    device = 'cuda:1'
    hidden_sizes = [64]
    lr = 3e-4
    repeat_per_collect = 10
    episode_per_collect = 20
    batch_size = 256
    training_num = 1  # 40
    test_num = 1  # 20
    resume_path = '/root/data/cj/program/EpMineEnv/logs/EpMine_0505_random/ppo_v14/42/230514-020416/policy_best_1684041860.147892.pth'
    
    env, train_envs, test_envs = make_env(seed, training_num, test_num, True)
    buffer_size = episode_per_collect * env.max_episode_length
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # model
    preprocess = PreProcessNet(device).to(device)
    actor = ActorProb(
        preprocess,
        action_shape,
        hidden_sizes=hidden_sizes,
        unbounded=True,
        device=device,
    ).to(device)
    critic = Critic(
        preprocess, 
        hidden_sizes=hidden_sizes,
        device=device
    ).to(device)
    actor_critic = ActorCritic(actor, critic)

    # 测试
    # obs = {'image': torch.zeros(64, 4, 128, 128, 3, dtype=torch.float32, device=device), 
    #        'state': torch.zeros(64, 9, dtype=torch.float32, device=device)}
    # print(actor(obs))
    # print(critic(obs))
    # raise Exception('Test passed.')
    
    # 网络正交初始化
    # torch.nn.init.constant_(actor.sigma_param, -0.5)
    # for m in actor_critic.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
    #         torch.nn.init.zeros_(m.bias)
    # for m in actor.mu.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.zeros_(m.bias)
    #         m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=0.9,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.25,
        ent_coef=0.0,
        reward_normalization=True,
        action_scaling=True,
        action_bound_method="clip",
        lr_scheduler=None,
        action_space=env.action_space,
        eps_clip=0.2,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=True,
    )

    # load a previous policy
    ckpt = torch.load(resume_path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    train_envs.set_obs_rms(ckpt["obs_rms"])
    test_envs.set_obs_rms(ckpt["obs_rms"])
    print("Loaded agent from: ", resume_path)

    # collector
    buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "ppo_v15"
    log_name = os.path.join('EpMine_0505_random', algo_name, str(seed), now)
    log_path = os.path.join('./logs/', log_name)

    # logger
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy, name):
        save_path = os.path.join(log_path, name + '.pth')
        state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, save_path)
        return save_path
    
    def save_best_fn(policy):
        save_fn(policy, f"policy_best_{time()}")

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=20,
        step_per_epoch=1e5,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        episode_per_collect=episode_per_collect,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )
    print(result)
