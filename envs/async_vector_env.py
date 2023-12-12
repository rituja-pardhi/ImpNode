from multiprocessing.pool import ThreadPool as Pool
from typing import Iterable, Callable, List, Optional, Union, Dict
import gymnasium as gym
import networkx as nx
import numpy as np
import time


class GraphAsyncVectorEnv:
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

    def step(self, action) -> (List[nx.Graph], np.array, np.array, List[Dict]):
        assert len(action) == self.num_envs
        obs_list = [None] * self.num_envs
        rew_arr = np.zeros(self.num_envs)
        truncated_arr = np.zeros(self.num_envs)
        terminal_arr = np.zeros(self.num_envs)
        info_list = []

        # Create a multiprocessing pool with as many processes as self.num_envs
        with Pool() as pool:
            start = time.time()
            results = pool.starmap(step_env, [(index, a, self.envs[index]) for index, a in enumerate(action)],
                                   chunksize=2)

            for index, observation, reward, terminated, truncated, info in results:

                obs_list[index] = observation
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


def step_env(index, action, env):
    # print(f"Before step of env {index}")
    observation, reward, terminated, truncated, info = env.step(action)
    # print(f"After step of env {index}")
    return index, observation, reward, terminated, truncated, info
