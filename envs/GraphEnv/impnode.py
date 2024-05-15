import random
from typing import Any, Union

import gymnasium as gym
import networkx as nx
import copy

from gymnasium.core import ActType
from matplotlib import pyplot as plt
from networkx import DiGraph
import numpy as np
from envs.spaces import GraphSpace


class ImpnodeEnv(gym.Env):

    def __init__(self, anc, g_type=None, num_nodes=None, mode=None,
                 data_path=None, file_name=None, render=False, max_removed_nodes=None):

        self.anc = anc
        self.g_type = g_type
        self.num_nodes = num_nodes

        self.mode = mode
        self.data_path = data_path
        self.file_name = file_name
        self.render = render
        self.max_removed_nodes = max_removed_nodes

        self.graph = None
        self.removed_nodes = None
        self.pos = None
        self.node_action_mask = None
        self.graph_len = None
        self.total_weight = None
        self.weights = None

        self.observation_space: Union[GraphSpace, None] = None
        self.action_space = None
        if not self.mode == 'test_multiple':
            self.setup()
        if self.render:
            self.render_graph()

    def setup(self, ep=0):

        self.graph, self.weights = self.gen_graph(ep)
        self.total_weight = sum(self.weights.values())
        # self.pos = nx.spring_layout(self.graph)
        self.graph_len = len(self.graph.nodes)
        self.observation_space = GraphSpace(num_nodes=int(self.graph_len))
        self.action_space = gym.spaces.Discrete(int(self.graph_len))

        # node action mask = [1,1,1,1,..num nodes]
        self.node_action_mask = np.ones((int(self.graph_len)), dtype=np.int8)
        self.removed_nodes = []
        obs, info = self._get_obs()

        return obs, info

    def get_degree_weights(self, graph):
        degrees = dict(graph.degree())
        max_degree = max(degrees.values())
        degree_weights = {int(node): degree / max_degree for node, degree in degrees.items()}
        return degree_weights

    def get_random_weights(self, graph):

        # degree_weight = self.get_degree_weights(graph)
        # delta = np.random.normal(0, 1)  # Random variable drawn from a normal distribution
        # median_degree = np.median(list(degree_weight.values()))
        # err = median_degree * delta
        # random_weights = {int(node): 0.5 * (degree + err) for node, degree in degree_weight.items()}

        degree = nx.degree(graph)
        # maxDegree = max(dict(degree).values())
        mu = np.mean(list(dict(degree).values()))
        std = np.std(list(dict(degree).values()))

        weights = {}
        for node in graph.nodes():
            episilon = np.random.normal(mu, std, 1)[0]
            weights[node] = 0.5 * degree[node] + episilon
            if weights[node] < 0.0:
                weights[node] = -weights[node]
        maxDegree = max(weights.values())
        for node in graph.nodes():
            weights[node] = weights[node] / maxDegree

        return weights

    def _get_obs(self) -> tuple[Any, dict[Any, Any]]:
        info = {
            'node_action_mask': self.node_action_mask
        }
        return self.graph, info

    def render_graph(self):
        # TODO remove node as well.. currently only edges removed
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)
        nx.draw(self.graph, self.pos, with_labels=True)
        return fig

    def step(self, actions: ActType) -> tuple[DiGraph, float | Any, bool, bool, dict]:
        assert not self._is_terminated(), "Env is terminated. Use reset()"

        if self.mode == 'test_multiple':
            actions = actions.to('cpu')
            nodes = actions.tolist()
            self.node_action_mask[nodes] = 0
            self.removed_nodes.extend(nodes)
            edges_to_remove = [i for i in self.graph.edges for node in nodes if int(i[0]) == int(node) or int(i[1]) == int(node)]
            self.graph.remove_edges_from(edges_to_remove)

        else:
            node = actions
            self.node_action_mask[actions] = 0
            self.removed_nodes.append(node)

            # remove edges from graph
            [self.graph.remove_edge(*i) for i in self.graph.edges if int(i[0]) == int(node) or int(i[1]) == int(node)]

        if self.render:
            self.render_graph()

        observation, info = self._get_obs()
        observation = copy.deepcopy(observation)
        reward = self._calculate_reward()

        terminated = self._is_terminated()
        truncated = False

        return observation, reward, terminated, truncated, info

    def _is_terminated(self):
        if self.max_removed_nodes:
            return len(self.graph.edges) == 0 or len(self.removed_nodes) >= self.max_removed_nodes
        return len(self.graph.edges) == 0

    def _calculate_reward(self):
        if self.mode == 'test_multiple':
            return 0
        elif self.mode == 'test':
            return self.connectivity()
        anc = -self.connectivity()
        return anc

    def reset(self, ep=0, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        Any, dict[Any, Any]]:

        obs, info = self.setup(ep)
        obs = copy.deepcopy(obs)
        return obs, info

    def gen_graph(self, ep):
        graph = nx.Graph()
        weights = {}
        if self.data_path:
            if not self.file_name:
                file_name = f"g_{ep}"
                graph = nx.read_gml(self.data_path / file_name)
            else:
                graph = nx.read_gml(self.data_path / self.file_name, destringizer=int)

            mapping = {node: int(node) for i, node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)
            weights = nx.get_node_attributes(graph, 'weight')
        else:
            if self.g_type == 'erdos-renyi':
                graph = nx.erdos_renyi_graph(n=random.randint(*self.num_nodes), p=0.15)
            elif self.g_type == 'powerlaw':
                graph = nx.powerlaw_cluster_graph(n=random.randint(*self.num_nodes), m=4, p=0.05)
            elif self.g_type == 'watts-strogatz':
                graph = nx.connected_watts_strogatz_graph(n=random.randint(*self.num_nodes), k=8, p=0.1)
            elif self.g_type == 'barabasi-albert':
                #m = random.randint(1, 5)
                m=4
                graph = nx.barabasi_albert_graph(n=random.randint(*self.num_nodes), m=m)
            else:
                print('Unknown graph type')

            if self.anc == 'dw_cn' or self.anc == 'dw_nd':
                weights = self.get_degree_weights(graph)
                nx.set_node_attributes(graph, weights, 'weight')
            elif self.anc == 'rw_cn' or self.anc == 'rw_nd':
                weights = self.get_random_weights(graph)
                nx.set_node_attributes(graph, weights, 'weight')
            elif self.anc == 'cn' or self.anc == 'nd':
                nx.set_node_attributes(graph, 1, 'weight')

            #nx.set_node_attributes(graph, weights, 'weight')
        nx.set_node_attributes(graph, 1, 'features')
        return graph, weights

    def connectivity(self):

        GCC = sorted(nx.connected_components(self.graph), key=len, reverse=True)

        if self.anc == 'cn':
            denominator = ((self.graph_len * (self.graph_len - 1)) / 2) * self.graph_len
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return sum(cn) / denominator

        elif self.anc == 'dw_cn':
            denominator = (self.graph_len * (self.graph_len - 1)) / 2
            weight = self.weights[self.removed_nodes[-1]] / self.total_weight
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return (sum(cn) * weight) / denominator

        elif self.anc == 'rw_cn':
            denominator = (self.graph_len * (self.graph_len - 1)) / 2
            weight = self.weights[self.removed_nodes[-1]] / self.total_weight
            cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
            return (sum(cn) * weight) / denominator

        elif self.anc == 'nd':
            denominator = self.graph_len * self.graph_len
            return len(GCC[0]) / denominator

        elif self.anc == 'dw_nd':
            denominator = self.graph_len
            weight = self.weights[int(self.removed_nodes[-1])] / self.total_weight
            return (len(GCC[0]) * weight) / denominator

        else:
            denominator = self.graph_len
            weight = self.weights[self.removed_nodes[-1]] / self.total_weight
            return (len(GCC[0]) * weight) / denominator
