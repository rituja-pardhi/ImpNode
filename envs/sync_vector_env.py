from typing import Iterable, Callable, List, Optional, Union, Dict
import gymnasium as gym
import networkx as nx
import numpy as np


class GraphSyncVectorEnv:
    def __init__(
            self,
            envs: List[gym.Env]
    ):
        self.envs = envs
        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata

    def reset(self):
        obs_list: List[nx.Graph] = []
        info_list: List = []
        for i in range(self.num_envs):
            obs, info = self.envs[i].reset()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions):
        assert len(actions) == self.num_envs
        obs_list: List[nx.Graph] = []
        rew_arr = np.zeros(self.num_envs)
        truncated_arr = np.zeros(self.num_envs)
        terminal_arr = np.zeros(self.num_envs)
        info_list = []
        for index, a in enumerate(actions):
            observation, reward, terminated, truncated, info = self.envs[index].step(a)
            obs_list.append(observation)
            rew_arr[index] = reward
            terminal_arr[index] = terminated
            info_list.append(info)
        return obs_list, rew_arr, terminal_arr, truncated_arr, info_list

    def action_space(self, env_index):
        return self.envs[env_index].action_space

    def observation_space(self, env_index):
        return self.envs[env_index].observation_space

    def num_nodes(self, env_index):
        return self.envs[env_index].num_nodes()

