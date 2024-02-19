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

    def __init__(self, anc, ba_nodes, ba_edges, max_removed_nodes, seed, render_option, data, data_path, train_mode):

        self.anc = anc
        self.ba_nodes = ba_nodes
        self.ba_edges = ba_edges
        self.max_removed_nodes = max_removed_nodes
        self.seed = seed
        self.render_option = render_option
        self.data = data
        self.data_path = data_path
        self.train_mode = train_mode

        self.graph = None
        self.removed_nodes = None
        self.pos = None
        self.degree_weight = None
        self.random_weight = None
        self.node_action_mask = None
        self.graph_len = None

        self.observation_space: Union[GraphSpace, None] = None

        self.setup()

        if self.render_option:
            self.render()

    def setup(self, ep=0):

        # make barabasi albert graph and add vector of ones as node features with size 5
        self.graph = self.gen_graph(ep)
        self.pos = nx.spring_layout(self.graph)

        self.graph_len = len(self.graph.nodes)

        self.degree_weight = self.normalized_degrees()
        self.random_weight = self.calculate_cost()

        self.observation_space = GraphSpace(num_nodes=int(len(self.graph.nodes)))
        self.action_space = gym.spaces.Discrete(int(len(self.graph.nodes)))

        # node action mask = [1,1,1,1,..num nodes]
        self.node_action_mask = np.ones((int(len(self.graph.nodes))), dtype=np.int8)

        self.removed_nodes = []

        obs, info = self._get_obs()

        return obs, info

    def normalized_degrees(self):
        degrees = dict(self.graph.degree())
        total_degree = sum(degrees.values())

        normalized_degrees = {int(node): degree / total_degree for node, degree in degrees.items()}

        return normalized_degrees

    def calculate_cost(self):
        delta = np.random.normal(0, 1)  # Random variable drawn from a normal distribution
        median_degree = np.median(list(self.degree_weight.values()))
        err = median_degree * delta
        cost = {int(node): 0.5 * (degree + err) for node, degree in self.degree_weight.items()}
        return cost

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

        # remove edges from graph
        [self.graph.remove_edge(*i) for i in self.graph.edges if int(i[0]) == int(node) or int(i[1]) == int(node)]

        if self.render_option:
            self.render()

        observation, info = self._get_obs()
        observation = copy.deepcopy(observation)
        reward = self._calculate_reward()

        terminated = self._is_terminated()
        truncated = False
        return observation, reward, terminated, truncated, info

    def _is_terminated(self):
        # if len(self.graph.edges) == 0:
        #     print('Graph is fully disconnected')
        return len(self.removed_nodes) >= self.max_removed_nodes or len(self.graph.edges) == 0

        #return len(self.graph.edges) == 0

    def _calculate_reward(self):

        if not self.train_mode:
            return self.connectivity()

        anc = -self.connectivity()
        return anc

    def reset(self, ep=0, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        Any, dict[Any, Any]]:
        obs, info = self.setup(ep)
        obs = copy.deepcopy(obs)
        return obs, info

    def gen_graph(self, ep):
        # graph = nx.barabasi_albert_graph(random.randint(*self.ba_nodes) * 2, self.ba_edges, self.seed)

        file_name = f"g_{ep}.gml"
        if self.data:
            graph = nx.read_gml(self.data_path / file_name)

        nx.set_node_attributes(graph, np.ones(5, dtype=int), 'features')
        return graph

    def connectivity(self):

        GCC = sorted(nx.connected_components(self.graph), key=len, reverse=True)

        if self.anc == 'cn':
            denominator = ((self.graph_len * (self.graph_len - 1)) / 2) * self.graph_len
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return sum(cn) / denominator

        elif self.anc == 'dw_cn':
            denominator = (self.graph_len * (self.graph_len - 1)) / 2
            weight = self.degree_weight[self.removed_nodes[-1]]
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return (sum(cn) * weight) / denominator

        elif self.anc == 'rw_cn':
            denominator = (self.graph_len * (self.graph_len - 1)) / 2
            weight = self.random_weight[self.removed_nodes[-1]]
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return (sum(cn) * weight) / denominator

        elif self.anc == 'nd':
            denominator = self.graph_len * self.graph_len
            return len(GCC[0]) / denominator

        elif self.anc == 'dw_nd':
            denominator = self.graph_len
            weight = self.degree_weight[int(self.removed_nodes[-1])]
            return (len(GCC[0]) * weight) / denominator

        else:
            denominator = self.graph_len
            weight = self.random_weight[self.removed_nodes[-1]]
            return (len(GCC[0]) * weight) / denominator
