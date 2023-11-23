from multiprocessing.pool import ThreadPool as Pool
from typing import Iterable, Callable, List, Optional, Union, Dict
import gymnasium as gym
import networkx as nx
import numpy as np


class GraphAsyncVectorEnv2:
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

    def step(self, action):
        assert len(action) == self.num_envs
        obs_list: List[nx.Graph] = []
        rew_arr = np.zeros(self.num_envs)
        truncated_arr = np.zeros(self.num_envs)
        terminal_arr = np.zeros(self.num_envs)
        info_list = []

        chunk_size = 8

        action_chunks = [action[i:i + chunk_size] for i in range(0, self.num_envs, chunk_size)]
        env_chunks = [self.envs[i:i + chunk_size] for i in range(0, self.num_envs, chunk_size)]
        indices_chunks = [(i, i + chunk_size) for i in range(0, self.num_envs, chunk_size)]

        num_chunks = len(indices_chunks)
        # Create a multiprocessing pool with as many processes as self.num_envs
        with Pool(processes=2) as pool:
            results = pool.starmap(step_multiple_envs,
                                   [(indices_chunks[i], action_chunks[i], env_chunks[i]) for i in range(num_chunks)])
            for indices, obs, rewards, terminals, _, infos in results:
                obs_list.extend(obs)
                rew_arr[indices[0]:indices[1]] = rewards
                terminal_arr[indices[0]:indices[1]] = terminals
                info_list.extend(infos)

        assert (len(obs_list) == self.num_envs)
        return obs_list, rew_arr, terminal_arr, truncated_arr, info_list

    def action_space(self, env_index):
        return self.envs[env_index].action_space

    def observation_space(self, env_index):
        return self.envs[env_index].observation_space

    def num_nodes(self, env_index):
        return self.envs[env_index].num_nodes()

def step_multiple_envs(indices, actions, envs):
    num_envs = len(envs)
    assert len(actions) == num_envs
    obs_list: List[nx.Graph] = []
    rew_arr = np.zeros(num_envs)
    truncated_arr = np.zeros(num_envs)
    terminal_arr = np.zeros(num_envs)
    info_list = []
    for index, a in enumerate(actions):
        observation, reward, terminated, truncated, info = envs[index].step(a)
        obs_list.append(observation)
        rew_arr[index] = reward
        terminal_arr[index] = terminated
        info_list.append(info)
    return indices, obs_list, rew_arr, terminal_arr, truncated_arr, info_list
