import time
from typing import Tuple, Dict, Any, Union, Set

import gymnasium as gym
import networkx as nx
import copy
import random

from gymnasium.core import ActType, ObsType
from matplotlib import pyplot as plt
from networkx import DiGraph
import numpy as np
from .spaces import GraphSpace


class ImpnodeEnv(gym.Env):

    def __init__(self, ba_edges, max_removed_nodes, seed, render_option, data, train_mode):

        self.ba_nodes = 30 #random.randint(15, 25)*2
        self.ba_edges = ba_edges
        self.max_removed_nodes = max_removed_nodes
        self.seed = seed
        self.render_option = render_option
        self.data = data
        self.train_mode = train_mode

        self.graph = None
        self.edge_list = None
        self.removed_nodes = None
        self.pos = None
        self.node_action_mask = None
        self.observation_space: Union[GraphSpace, None] = None

        self.nd_denominator = None
        self.cn_denominator = None

        self.setup()

        if self.render_option:
            self.render()

    def setup(self, ep=0):

        # make barabasi albert graph and add vector of ones as node features with size 5
        self.graph = self.gen_graph(ep)
        self.pos = nx.spring_layout(self.graph)

        # store denominator values according to original graph
        self.nd_denominator = int(len(self.graph.nodes))
        self.cn_denominator = (int(len(self.graph.nodes)) * (int(len(self.graph.nodes)) - 1)) / 2

        self.observation_space = GraphSpace(num_nodes=int(len(self.graph.nodes)))
        self.action_space = gym.spaces.Discrete(int(len(self.graph.nodes)))

        # node action mask = [1,1,1,1,..num nodes]
        self.node_action_mask = np.ones((int(len(self.graph.nodes))), dtype=np.int8)

        self.removed_nodes = []
        self.edge_list = list(nx.to_edgelist(self.graph))
        obs, info = self._get_obs()

        return obs, info

    def _get_obs(self) -> tuple[Any, dict[Any, Any]]:
        info = {
            'node_action_mask': self.node_action_mask
        }
        return self.graph, info

    def render(self):
        # TODO remove node as well.. currently only edges removed
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)
        nx.draw(self.graph, self.pos, with_labels=True)
        return fig

    def step(self, action: ActType) -> tuple[DiGraph, float | Any, bool, bool, dict]:
        assert not self._is_terminated(), "Env is terminated. Use reset()"

        node = action
        self.node_action_mask[action] = 0
        self.removed_nodes.append(node)

        # reward calculation requires cn and nd of graph before removing the node (prev)
        Gcc_prev = sorted(nx.connected_components(nx.Graph(self.edge_list)), key=len, reverse=True)
        gcc_prev_lengths = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in Gcc_prev]
        cn_prev = sum(gcc_prev_lengths)
        nd_prev = len(Gcc_prev[0])

        # remove edges from graph and edge list
        [self.graph.remove_edge(*i) for i in self.graph.edges if i[0] == node or i[1] == node]
        [self.edge_list.remove(i) for i in self.edge_list if i[0] == node or i[1] == node]

        if self.render_option:
            self.render()

        observation, info = self._get_obs()
        observation = copy.deepcopy(observation)
        reward = self._calculate_reward(nd_prev, cn_prev)

        terminated = self._is_terminated()
        truncated = False
        return observation, reward, terminated, truncated, info

    def _is_terminated(self):
        return len(self.removed_nodes) >= self.max_removed_nodes

    def _calculate_reward(self, nd_prev, cn_prev):
        Gcc_current = sorted(nx.connected_components(nx.Graph(self.edge_list)), key=len, reverse=True)
        gcc_current_lengths = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in Gcc_current]
        sum_gcc_current = sum(gcc_current_lengths)

        nd = (nd_prev - len(Gcc_current[0])) / self.nd_denominator
        cn = (cn_prev - sum_gcc_current) / self.cn_denominator
        if not self.train_mode:
            return sum_gcc_current / self.cn_denominator
        return cn

    def reset(self, ep=0, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        Any, dict[Any, Any]]:
        obs, info = self.setup(ep)
        obs = copy.deepcopy(obs)
        return obs, info

    def gen_graph(self, ep):
        graph = nx.barabasi_albert_graph(self.ba_nodes, self.ba_edges, self.seed)

        if self.data:
            graph = nx.read_gml("C:/Users/rituja.pardhi/Thesis/ma-rituja-pardhi/DQN_trial/data/synthetic/uniform_cost"
                                "/30-50/g_{}".format(ep))

        nx.set_node_attributes(graph, np.ones(5, dtype=int), 'features')
        return graph
