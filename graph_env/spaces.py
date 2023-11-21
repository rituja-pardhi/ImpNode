import typing

import gymnasium as gym
import networkx as nx
import numpy as np


class GraphSpace(gym.Space[nx.Graph]):
    def __init__(self, num_nodes: int, dtype=np.int64, seed=None):
        self.num_nodes = num_nodes

        super().__init__((num_nodes, num_nodes), dtype, seed)

    def contains(self, x) -> bool:
        return type(x) == nx.Graph and len(x.nodes) == self.num_nodes

    def __repr__(self):
        return f"DiGraph({self.num_nodes})"
