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


def make_env(seed: int = 0, test_num: int = 4, no_graph: bool = True):
    env = ShmemVectorEnv([lambda: EpMineEnv(
        port=5000, width=400, height=200, 
        no_graph=no_graph, only_state=False, only_image=True) for _ in range(test_num)]
    )
    env.seed(seed)
    return env


if __name__ == "__main__":
    resume_path = '/root/data/cj/program/EpMineEnv/logs/EpMine/ppo_v7/42/230509-121505/policy_best.pth'
    test_num = 4
    test_episodes = 20
    no_graph = False

    seed = 42
    device = 'cuda:1'
    hidden_sizes = [128, 64]
    lr = 3e-4
    env = make_env(seed, test_num, no_graph)
    state_shape = env.observation_space[0].shape or env.observation_space[0].n
    action_shape = env.action_space[0].shape or env.action_space[0].n
    max_action = env.action_space[0].high[0]

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

    # load a previous policy
    policy.load_state_dict(torch.load(resume_path, map_location=device))
    print("Loaded agent from: ", resume_path)

    # collector
    policy.eval()
    test_collector = Collector(policy, env)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_episodes)
    print(result)
