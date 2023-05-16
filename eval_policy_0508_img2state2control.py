import random

import numpy as np
import tianshou as ts
import torch
from envs.SingleAgent.mine_toy_0508_img2state2control import EpMineEnv
from resnet import Img_model
from tianshou.data import Batch, Collector
from tianshou.env import VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch import nn
from torch.distributions import Independent, Normal
from tqdm import tqdm
from tianshou.utils import RunningMeanStd


class Seven2Nine(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.mine_position = torch.tensor(
            [-5.02311832e-07, 7.00004572e-02, -8.80874085e-07],
            dtype=torch.float32,
            device=device
        )
        self.device = device
        self.obs_rms = RunningMeanStd()
    
    def forward(self, x):
        dist_to_mine = torch.sqrt(torch.pow(x[:,-3:]-self.mine_position, 2).sum(axis=1))
        angel_to_mine = torch.arctan((self.mine_position[2] - x[:,-1]) / (self.mine_position[0] - x[:,-3] + 1e-8))
        out = torch.cat([x, dist_to_mine.unsqueeze(1), angel_to_mine.unsqueeze(1)], axis=1).to('cpu')
        out = self.obs_rms.norm(out.detach().numpy())
        out = torch.as_tensor(out, dtype=torch.float32, device=self.device)
        return out


class SequentialWrapper(nn.Module):
    def __init__(self, sequence) -> None:
        super().__init__()
        self.sequential = sequence
    
    def forward(self, x, state=None, info={}):
        return self.sequential(x)
    

def make_env(seed: int = 0, test_num: int = 4, no_graph: bool = True):
    file_name = '/root/data/cj/program/EpMineEnv/envs/SingleAgent/MineField_Linux-0508-big-image-high-freq/drl.x86_64'
    env = ts.env.ShmemVectorEnv(
        [lambda: EpMineEnv(
        file_name=file_name, port=6000, 
        width=400, height=200,
        max_episode_steps=200,
        no_graph=no_graph, only_state=False, only_image=True) for _ in range(test_num)]
    )
    env.seed(seed)
    return env


if __name__ == "__main__":
    resume_path = '/root/data/cj/program/EpMineEnv/logs/EpMine_0505_random/ppo_v9/42/230511-034009/policy_best_672.pth'
    test_num = 1
    test_episode = 5
    no_graph = False
    
    seed = 42
    device = 'cuda:1'
    hidden_sizes = [64, 64]
    lr = 3e-4

    model = Img_model(512)
    model.load_state_dict(torch.load('./logs/resnet/default_dict.pth'))

    env = make_env(seed, test_num, no_graph)
    state_shape = 9
    action_shape = env.action_space[0].shape or env.action_space[0].n
    max_action = env.action_space[0].high[0]

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # load a previous policy
    ckpt = torch.load(resume_path, map_location=device)

    # model
    test = torch.zeros((1,224,224,3), dtype=torch.float32, device=device)
    net_a_feature = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
        device=device
    )
    net_a_feature.load_state_dict({i:ckpt['model']['actor.preprocess.'+i] for i in net_a_feature.state_dict().keys() if 'actor.preprocess.'+i in ckpt['model']})
    seven2nine = Seven2Nine(device)
    seven2nine.obs_rms = ckpt['obs_rms']
    net_a = SequentialWrapper(
        nn.Sequential(
            model,
            seven2nine,
            net_a_feature,
        ).to(device)
    )
    print(net_a(test))
    actor = ActorProb(
        net_a,
        action_shape,
        unbounded=True,
        device=device,
        preprocess_net_output_dim=hidden_sizes[-1]
    ).to(device)
    actor.sigma_param.data = ckpt['model']['actor.sigma_param']
    actor.mu.load_state_dict({i:ckpt['model']['actor.mu.'+i] for i in actor.mu.state_dict().keys() if 'actor.mu.'+i in ckpt['model']})
    print(actor(test))

    net_c_feature = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
        device=device
    )
    net_c_feature.load_state_dict({i:ckpt['model']['critic.preprocess.'+i] for i in net_c_feature.state_dict().keys() if 'critic.preprocess.'+i in ckpt['model']})
    net_c = SequentialWrapper(
        nn.Sequential(
            model,
            seven2nine,
            net_c_feature,
        ).to(device)
    )
    critic = Critic(
        net_c, 
        device=device,
        preprocess_net_output_dim=hidden_sizes[-1]  
    ).to(device)
    critic.last.load_state_dict({i:ckpt['model']['critic.last.'+i] for i in critic.last.state_dict().keys() if 'critic.last.'+i in ckpt['model']})
    actor_critic = ActorCritic(actor, critic)

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
        action_space=env.action_space[0],
        eps_clip=0.2,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=True,
    )

    policy.eval()

    obs, _ = env.reset()
    done = False
    while not done:
        pass
    # test_collector = Collector(policy, env)
    # test_collector.reset()
    # result = test_collector.collect(n_episode=test_episode)
    # print(result)
    # print()
    # print('Success rate: ', result['success_rate'])
    # success_per_seed = [0] * 10
    # episode_per_seed = [0] * 10
    # for i in range(len(result['idxs'])):
    #     episode_per_seed[result['idxs'][i]] += 1
    #     if result['lens'][i] < 202:
    #         success_per_seed[result['idxs'][i]] += 1

    # print('episode per seed: ', episode_per_seed)
    # print('success rate per seed: ', np.array(success_per_seed) / np.array(episode_per_seed))
