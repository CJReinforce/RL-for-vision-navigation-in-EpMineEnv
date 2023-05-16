import numpy as np
import tianshou as ts
import torch
import random
from envs.SingleAgent.mine_toy import EpMineEnv
from tianshou.data import Batch, Collector
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.discrete import Critic
from torch import nn
from torch.distributions import Independent, Normal
from tqdm import tqdm
from train_ppo import PreProcessNet, VectorMineEnvNormObs


def make_env(seed: int, test_num: int, no_graph: bool = True):
    file_name = '/root/data/cj/program/EpMineEnv/envs/SingleAgent/MineField_Linux-0510-random/drl.x86_64'
    env = ShmemVectorEnv(
        [lambda: EpMineEnv(file_name=file_name, port=7000,
                           no_graph=no_graph, only_state=False, only_image=False) for _ in range(test_num)]
    )
    env.seed(seed)
    env = VectorMineEnvNormObs(env, update_obs_rms=False)
    return env


if __name__ == "__main__":
    resume_path = '/root/data/cj/program/EpMineEnv/logs/EpMine_0505_random/ppo_v14/42/230514-020416/policy_best_1684041860.147892.pth'
    test_num = 9
    test_episodes = 100
    no_graph = False

    seed = 42
    device = 'cuda:1'
    hidden_sizes = [64]
    lr = 3e-4
    env = make_env(seed, test_num, no_graph)
    action_shape = env.action_space[0].shape or env.action_space[0].n
    max_action = env.action_space[0].high[0]

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

    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=0.99,
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
    ckpt = torch.load(resume_path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    env.set_obs_rms(ckpt["obs_rms"])
    print("Loaded agent from: ", resume_path)

    policy.eval()
    test_collector = Collector(policy, env)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_episodes)
    print(result)
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

    
    # obs, _ = env.reset()
    # original_reward_list = []
    # dense_reward_list = []
    # step_list = []

    # for _ in tqdm(range(10)):
    #     original_reward = 0.0
    #     dense_reward = 0.0
    #     step = 0
    #     done = False
    #     while not done:
    #         obs = torch.as_tensor(obs).float().to(device)
    #         batch = Batch({'obs': obs, 'info': {}})
    #         result = policy(batch, None)
    #         action = result.act.cpu()
    #         # print(action)
    #         obs, reward, done, truncate, info = env.step(action)
    #         # print(info)
    #         dense_reward += reward[0]
    #         original_reward += info[0]['original_reward']
    #         step += 1
    #         # break
    #     original_reward_list.append(original_reward)
    #     dense_reward_list.append(dense_reward)
    #     step_list.append(step)
    #     obs, _ = env.reset()
    
    # print('original reward: ', original_reward_list)
    # print('dense reward: ', dense_reward_list)
    # print('step: ', step_list)
