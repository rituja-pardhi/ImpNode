import time
import warnings
from typing import Tuple, Dict, Any, Union, Set

import gymnasium as gym
import networkx as nx
import copy
import random

from gymnasium.core import ActType, ObsType
from matplotlib import pyplot as plt
from networkx import DiGraph
import numpy as np
from envs.spaces import GraphSpace


class GraphNavEnv(gym.Env):
    def __init__(self, fix_random_graphs=False):

        self.graph = None
        self.current_node = None
        self.goal_node = None
        self._max_edge_length = 10
        self.fix_random_graphs = fix_random_graphs
        self.init_start_node = None
        self.observation_space: Union[GraphSpace, None] = None
        self.action_space = None
        self._build_graph()

    def _build_graph(self):
        if self.graph is not None and self.fix_random_graphs:
            self.current_node = self.init_start_node
            return

        reachable_goal = False

        while not reachable_goal:

            self.graph = nx.erdos_renyi_graph(n=10, p=0.5)
            for _, _, d in self.graph.edges(data=True):
                d['length'] = np.random.randint(0, self._max_edge_length)
            self.pos = nx.spring_layout(self.graph)

            # Assign features to nodes: 0 for all nodes initially
            nx.set_node_attributes(self.graph, 0, 'is_agent')
            nx.set_node_attributes(self.graph, 0, 'is_goal')

            # Randomly select start and goal nodes
            self.current_node, self.goal_node = np.random.choice(self.graph.nodes, 2, replace=False)
            self.init_start_node = self.current_node
            self.graph.nodes[self.current_node]['is_agent'] = 1
            self.graph.nodes[self.goal_node]['is_goal'] = 1

            # Make sure that there is actually a path between start and goal node
            if nx.has_path(self.graph, self.current_node, self.goal_node):
                reachable_goal = True

        # Update action space based on the number of nodes
        self.observation_space = GraphSpace(num_nodes=int(len(self.graph.nodes)))
        self.action_space = gym.spaces.Discrete(len(self.graph.nodes))

    def reset(self, ep=None, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        nx.Graph, dict[str, Any]]:
        self._build_graph()

        info = self._get_info()
        return self._get_observation(), info

    def _get_info(self):
        action_mask = self._get_action_mask()
        info = {"node_action_mask": action_mask, "goal_node": self.goal_node, "agent_node": self.current_node}
        return info

    def step(self, action):
        # Generate action mask for current node
        action_mask = self._get_action_mask()

        if action_mask[action] == 0:
            warnings.warn("Agent did invalid move")
            reward = -1000 / self._max_edge_length
        else:
            reward = -self.graph.edges[(self.current_node, action)]['length'] / self._max_edge_length

        # Move agent
        self.graph.nodes[self.current_node]['is_agent'] = 0
        self.current_node = action
        self.graph.nodes[self.current_node]['is_agent'] = 1

        done = self.current_node == self.goal_node
        truncated = False

        info = self._get_info()
        obs = self._get_observation()
        return obs, reward, done, truncated, info

    def sample_valid_action(self):
        action_mask = self._get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else None

    def _get_action_mask(self):
        action_mask = np.zeros(self.action_space.n, dtype=np.int8)#bool_)
        for neighbor in self.graph.neighbors(self.current_node):
            if neighbor != self.current_node:
                action_mask[neighbor] = 1
        return action_mask

    def _get_observation(self):
        return copy.deepcopy(self.graph)

    def render(self, mode='human'):
        color_map = []
        for node in self.graph:
            if self.graph.nodes[node]['is_agent']:
                color_map.append('black')  # Agent node
            elif self.graph.nodes[node]['is_goal']:
                color_map.append('red')  # Goal node
            else:
                color_map.append('blue')  # Other nodes

        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, self.pos, node_color=color_map, with_labels=False, node_size=700)
        nx.draw_networkx_labels(self.graph, self.pos, font_color='white')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=nx.get_edge_attributes(self.graph, 'length'))
        plt.show()


