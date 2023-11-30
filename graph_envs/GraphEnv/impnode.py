from typing import Tuple, Dict, Any, Union

import gymnasium as gym
import networkx as nx
import copy

from gymnasium.core import ActType, ObsType
from matplotlib import pyplot as plt
from networkx import DiGraph

from .spaces import GraphSpace


class ImpnodeEnv(gym.Env):

    def __init__(self, ba_nodes, ba_edges, max_removed_nodes, seed):
        self.nd_denominator = None
        self.cn_denominator = None
        self.graph = None

        self.ba_nodes = ba_nodes
        self.ba_edges = ba_edges
        self.removed_nodes = None
        self.seed = seed
        self.pos = None

        self.max_removed_nodes = max_removed_nodes

        self.observation_space: Union[GraphSpace, None] = None

        self.setup()
        # self.render()

    def setup(self):
        self.graph = nx.barabasi_albert_graph(self.ba_nodes, self.ba_edges, self.seed)
        self.pos = nx.spring_layout(self.graph)

        # store denominator values according to original graph
        self.nd_denominator = self.num_nodes()
        self.cn_denominator = (self.num_nodes() * (self.num_nodes() - 1)) / 2

        self.observation_space = GraphSpace(num_nodes=self.num_nodes())

        self.action_space = gym.spaces.Discrete(self.num_nodes())

        self.removed_nodes = []
        obs, info = self._get_obs()

        return obs, info

    def num_nodes(self):
        return int(len(self.graph.nodes))

    def _get_obs(self) -> Tuple[nx.DiGraph, Dict]:
        info = {
        }
        return self.graph, info

    def render(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)
        nx.draw(self.graph, self.pos, with_labels=True)
        return fig

    def step(self, action: ActType) -> tuple[DiGraph, float | Any, bool, bool, dict]:
        assert not self._is_terminated(), "Env is terminated. Use reset()"

        node = action

        self.removed_nodes.append(node)

        # prev_graph = copy.deepcopy(self.graph)
        Gcc_prev = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        gcc_prev_lengths = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in Gcc_prev]
        cn_prev = sum(gcc_prev_lengths)
        nd_prev = len(Gcc_prev[0])

        self.graph.remove_node(node)

        # self.render()

        observation, info = self._get_obs()
        observation = copy.deepcopy(observation)
        reward = self._calculate_reward(nd_prev, cn_prev)

        terminated = self._is_terminated()
        truncated = False
        return observation, reward, terminated, truncated, info

    def _is_terminated(self):
        return len(self.removed_nodes) >= self.max_removed_nodes

    def _calculate_reward(self, nd_prev, cn_prev):
        Gcc_current = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        gcc_current_lengths = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in Gcc_current]
        sum_gcc_current = sum(gcc_current_lengths)

        nd = (nd_prev - len(Gcc_current[0])) / self.nd_denominator
        cn = (cn_prev - sum_gcc_current) / self.cn_denominator

        return cn

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        obs, info = self.setup()
        obs = copy.deepcopy(obs)
        return obs, info
