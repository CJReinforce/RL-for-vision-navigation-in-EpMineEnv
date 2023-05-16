import os
import random
import socket
import time
from collections import deque
from typing import Optional

import cv2 as cv
import gymnasium as gym
import numpy as np
from mlagents_envs.base_env import ActionTuple, DecisionSteps
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel


def IsOpen(port, ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    result = s.connect_ex((ip,int(port)))
    if result == 0:
        print("port {} is used".format(port))
        return True
    else:
        print("port {} is not used".format(port))
        return False

TEAM_NAME = 'ControlEP?team=0'
AGENT_ID = 0

def warp_action(action):
    action_dict = {'{}_{}'.format(TEAM_NAME, AGENT_ID): action}
    return action_dict

class EpMineEnv(gym.Env):

    def __init__(self,
                 file_name: str = "/root/data/cj/program/EpMineEnv/envs/SingleAgent/MineField_Linux-0421/drl.x86_64",
                 port: Optional[int] = 2000,
                 seed: int = 0,
                 work_id: int = 0,
                 time_scale: float = 1.0,
                 max_episode_steps: int = 1000,
                 width=128, 
                 height=128,
                 only_image: bool = True,
                 only_state: bool = False,
                 no_graph: bool = True):
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(width=width, height=height,
                                                                  time_scale=time_scale)
        self._engine_Environment_channel = EnvironmentParametersChannel()
        self.env = None
        self.port = port
        self.work_id = work_id
        self.eng_conf_channel = engine_configuration_channel
        self.env_file_name = file_name
        self.sd = seed
        self.no_graph = no_graph
        self.max_episode_length = max_episode_steps
        self.step_num = 0
        self.only_image = only_image
        self.only_state = only_state
        self.last_dist = 0.0
        self.current_results = None
        self.catch_state = 0
        self.image_queue = deque(maxlen=8)
    
    def seed(self, sd=0):
        if self.env is not None:
            self.env.close()
        worker_id = sd
        while IsOpen(self.port+worker_id):
            worker_id += 1
        self.env = UnityEnvironment(file_name=self.env_file_name,
                                    base_port=self.port,
                                    seed=sd,
                                    worker_id=worker_id,
                                    side_channels=[self._engine_Environment_channel, self.eng_conf_channel],
                                    no_graphics=self.no_graph
                                    )

    @property
    def observation_space(self):
        state_space = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(7,), dtype=np.float32)
        if self.only_image:
            image_space = gym.spaces.Box(low=0, high=255, shape=(8, 128, 128, 3), dtype=np.uint8)
            return image_space
        elif self.only_state:
            return state_space
        

        return gym.spaces.Dict({'image': image_space, 'state': state_space})
    
    @property
    def action_space(self):
        con_spc = gym.spaces.Box(low=np.array([-10.0, -10.0, -3.0]), high=np.array([10.0, 10.0, 3.0]), shape=(3,), dtype=np.float32)
        return con_spc
    
    def decoder_results(self, results):
        org_obs = results[TEAM_NAME].obs
        img = cv.cvtColor(np.array(org_obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR)
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        arm_angle = org_obs[1][AGENT_ID][7]
        catching = org_obs[1][AGENT_ID][8]
        is_catched = org_obs[1][AGENT_ID][9]
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        state = org_obs[1][AGENT_ID]
        obs = {"image": img, "state": state}
        # print(position, mineral_pose)
        self.catch_state = catching
        if self.only_image:
            if len(self.image_queue) == 0:
                self.image_queue.extend([img] * 8)
            else:
                self.image_queue.append(img)
            assert len(self.image_queue) == 8
            return np.array(self.image_queue, dtype=np.uint8)
        elif self.only_state:
            return np.array(org_obs[1][AGENT_ID][:7])
        return obs
    
    def get_robot_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        return position, rotation
    
    def get_mine_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        return mineral_pose
    
    def get_dist_to_mine(self, reuslts):
        mine_pose = self.get_mine_pose(results=reuslts)
        robot_pose = self.get_robot_pose(results=reuslts)[0]
        # dist = np.sqrt(robot_pose[0] ** 2 + robot_pose[2] ** 2)
        dist = np.sqrt((robot_pose[0] - mine_pose[0]) ** 2 + (robot_pose[1] - mine_pose[1]) ** 2 + (robot_pose[2] - mine_pose[2]) ** 2)
        return dist
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.sd = seed
        if self.env is None:
            self.seed(self.sd)
        self.image_queue.clear()
        self.step_num = 0
        self.env.reset()
        obs, _, _, _, _ = self._step()
        self.last_dist = self.get_dist_to_mine(self.current_results)
        return obs, {}
    
    def get_reward(self, results):
        reward = results[TEAM_NAME].reward[AGENT_ID]
        return reward
    
    def get_dense_reward(self, results):
        final_reward = results[TEAM_NAME].reward[AGENT_ID]
        # print(final_reward)
        current_dist = self.get_dist_to_mine(reuslts=results)
        # print(self.last_dist - current_dist)
        delta_r = (self.last_dist - current_dist)
        final_reward += delta_r
        # if current_dist < 0.5:
        #     final_reward += (1-current_dist) / 0.5
        self.last_dist = current_dist
        return final_reward
    
    def step(self, action):
        # action: [vy, vx, vw, arm_ang, catching]
        action = [action[0], action[1], action[2], 10.0, 1.0]
        action = ActionTuple(np.array([action], dtype=np.float32))
        action_dict = warp_action(action=action)
        obs, reward, done, truncate, info = self._step(action_dict=action_dict)
        self.step_num += 1
        return obs, reward, done, truncate, info

    def _step(self, action_dict=None) -> DecisionSteps:
        all_agents = []
        for behavior_name in self.env.behavior_specs:
            # add actions
            for agent_id in self.env.get_steps(behavior_name)[0].agent_id:
                key = behavior_name + "_{}".format(agent_id)
                all_agents.append(key)
                if (action_dict != None):
                    self.env.set_action_for_agent(behavior_name, agent_id,
                                                  action_dict[key])
        self.env.step()

        decision_result = dict()
        terminal_result = dict()
        for behavior_name in self.env.behavior_specs:
            decision_result[behavior_name], terminal_result[behavior_name] = self.env.get_steps(behavior_name)
        done = False
        truncate = False
        info = {}
        reward = 0.0
        if len(terminal_result[TEAM_NAME]) != 0:
            done = True
            # info = terminal_result[TEAM_NAME]
            obs = self.decoder_results(results=terminal_result)
            reward = self.get_dense_reward(results=terminal_result)
            self.current_results = terminal_result
            robot_position = self.get_robot_pose(results=terminal_result)[0]
        else:
            # info = decision_result[behavior_name]
            obs = self.decoder_results(results=decision_result)
            reward = self.get_dense_reward(results=decision_result)
            self.current_results = decision_result
            robot_position = self.get_robot_pose(results=decision_result)[0]
        if self.step_num > self.max_episode_length:
            truncate = True

        info["robot_position"] = robot_position
        info["original_reward"] = self.get_reward(results=self.current_results)
        
        return obs, reward, done, truncate, info


def main():
    env = EpMineEnv(port=3000, no_graph=True, only_state=False, only_image=True)
    print('#'*50)
    print('Created the environment!')
    print('#'*50)
    # result = env.step()  # if not, the env won't update
    obs, _ = env.reset()
    print(obs.shape)
    done = False
    step = 0

    print('#'*50)
    print('Resetted the environment!')
    print('#'*50)

    while not done:
        action = env.action_space.sample()
        # print('action: ', action)
        obs, reward, done, truncate, info = env.step(action)
        print(obs.shape)
        position = info["robot_position"]
        # print('position: ', position)
        print('reward: ', reward)
        # print(np.array(obs["image"]).shape)
        # cv.imwrite("images/{}-({}, {}).png".format(step, position[0], position[2]), obs)
        print('----------------------------------------')
        step += 1
        # print(step)
        if step == 5:
            break


if __name__ == '__main__':
    main()
