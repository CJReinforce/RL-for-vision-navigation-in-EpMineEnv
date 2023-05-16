import datetime
import os

import gymnasium as gym
import numpy as np
import tianshou as ts
import torch
from envs.SingleAgent.mine_toy_consecutive_frame import EpMineEnv
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.discrete import Critic
from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
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
        self.device = device
        with torch.no_grad():
            self.output_dim = np.prod(self.forward(torch.zeros(1, 8, 128, 128, 3))[0].shape[1:])
            print(f'CNN output dimension: {self.output_dim}')
    
    def forward(self, obs, state=None, info={}):
        out = torch.as_tensor(obs, dtype=torch.float32)
        shape = out.shape
        if len(shape) != 5 or shape[1:] != (8, 128, 128, 3):
            raise Exception(f'Unknown obs shape: {shape}')
        out = out.reshape(-1, 128, 128, 3)
        out = out.permute(0, 3, 1, 2)
        out = out.to(device)
        out = self.net(out/255)  # obs normalization
        out = out.reshape(shape[0], -1)
        return out, state


def make_env(seed, training_num, test_num):
    env = EpMineEnv(no_graph=True, only_state=False, only_image=True)
    train_envs = ShmemVectorEnv(
        [lambda: EpMineEnv(port=3000, no_graph=True, only_state=False, only_image=True) for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv(
        [lambda: EpMineEnv(port=4000, no_graph=True, only_state=False, only_image=True) for _ in range(test_num)]
    )
    
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    
    return env, train_envs, test_envs


if __name__ == "__main__":
    seed = 42
    device = 'cuda:1'
    hidden_sizes = [128, 64]
    lr = 3e-4
    buffer_size = 2e4
    batch_size = 256
    training_num = 20
    test_num = 20
    # baseline_path = '/root/data/cj/program/EpMineEnv/logs/EpMine/ppo_v4/42/230508-065453/policy_best.pth'

    env, train_envs, test_envs = make_env(seed, training_num, test_num)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # model
    def layer_init(
        layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
    ) -> nn.Module:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    net = CNN(device=device)
    actor = ActorProb(
        net,
        action_shape,
        hidden_sizes=hidden_sizes,
        unbounded=True,
        device=device,
    ).to(device)
    critic = Critic(net, hidden_sizes, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    # print(actor_critic)

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
    if False:
        ckpt = torch.load(baseline_path, map_location=device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", baseline_path)

    # collector
    buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "ppo_v7"
    log_name = os.path.join('EpMine', algo_name, str(seed), now)
    log_path = os.path.join('./logs/', log_name)

    # logger
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy, name):
        save_path = os.path.join(log_path, name + '.pth')
        torch.save(policy.state_dict(), save_path)
        return save_path
    
    def save_best_fn(policy):
        save_fn(policy, "policy_best")

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=1e5,
        repeat_per_collect=5,
        episode_per_test=test_num,
        batch_size=batch_size,
        step_per_collect=2000,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )
    print(result)
