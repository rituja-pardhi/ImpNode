from envs.GraphEnv.impnode import ImpnodeEnv
from envs.sync_vector_env import GraphSyncVectorEnv
from envs.async_vector_env import GraphAsyncVectorEnv
import time
import numpy as np


class vector_envs_experiment:

    def __init__(self, ba_nodes, ba_edges, max_removed_nodes, seed, num_envs, num_episodes):
        self.ba_nodes = ba_nodes
        self.ba_edges = ba_edges
        self.max_removed_nodes = max_removed_nodes
        self.seed = seed
        self.num_envs = num_envs
        self.num_episodes = num_episodes

    def make_env_fn(self, seed):
        env = ImpnodeEnv(ba_nodes=self.ba_nodes, ba_edges=self.ba_edges, max_removed_nodes=self.max_removed_nodes,
                         seed=self.seed)
        return env

    def sync_func(self):
        sync_envs = GraphSyncVectorEnv([self.make_env_fn(i) for i in range(self.num_envs)])
        for i in range(self.num_envs):
            sync_envs.action_space(i).seed(i)

        for i in range(self.num_episodes):
            terminated = False
            num_nodes = [sync_envs.num_nodes(i) for i in range(self.num_envs)]
            masks = [np.ones(nodes, dtype=np.int8) for nodes in num_nodes]
            while not terminated:
                actions = [sync_envs.action_space(i).sample(mask=masks[i]) for i in range(self.num_envs)]
                for j in range(self.num_envs):
                    masks[j][actions[j]] = 0
                start = time.time()
                observation, reward, terminated, truncated, info = sync_envs.step(actions)
                end = time.time()
                print(end - start)
                terminated = sum(terminated)
            sync_envs.reset()

    def async_func(self):
        async_envs = GraphAsyncVectorEnv([self.make_env_fn(i) for i in range(self.num_envs)])
        for i in range(self.num_envs):
            async_envs.action_space(i).seed(i)

        for i in range(self.num_episodes):
            terminated = False
            num_nodes = [async_envs.num_nodes(i) for i in range(self.num_envs)]
            masks = [np.ones(nodes, dtype=np.int8) for nodes in num_nodes]
            while not terminated:
                actions = [async_envs.action_space(i).sample(mask=masks[i]) for i in range(self.num_envs)]
                for j in range(self.num_envs):
                    masks[j][actions[j]] = 0
                start = time.time()
                observation, reward, terminated, truncated, info = async_envs.step(actions)
                end = time.time()
                print(end - start)
                terminated = sum(terminated)
            async_envs.reset()
