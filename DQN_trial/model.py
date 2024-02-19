"""
Script that contains details about the neural network model used for the DQN Agent
"""

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.optim as optim


class SumAgg(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j


class DQNNet(nn.Module):
    """
    Class that defines the architecture of the neural network for the DQN agent
    """

    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()

        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size // 2)
        self.linear3 = nn.Linear(output_size, output_size // 2)
        self.sum_agg = SumAgg()

        self.dense1 = nn.Linear(2 * output_size, 64)
        self.dense2 = nn.Linear(64, 1)

        # self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, data, embedding=False):
        x, edge_index = data.features.to(torch.float32), data.edge_index

        x = F.relu(self.linear1(x))
        x = x / x.norm(dim=-1, keepdim=True)

        for _ in range(3):
            neighbor_messages = self.sum_agg(x, edge_index)

            x = F.relu(torch.cat([self.linear2(x), self.linear3(neighbor_messages)], dim=-1))
            x = x / x.norm(dim=-1, keepdim=True)
        if embedding:
            return x

        if len(data) == 3:  # no batch: for select action or test
            x = torch.cat((x[:-1], x[-1].repeat(len(x) - 1, 1)), dim=1)
        else:
            x = torch.cat([torch.cat(
                (x[data.batch == i][:-1], x[data.batch == i][-1].repeat(len(x[data.batch == i]) - 1, 1)), dim=1) for i
                in range(data.num_graphs)])

        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

    def save_model(self, filename):
        """
        Function to save model parameters

        Parameters
        ---
        filename: str
            Location of the file where the model is to be saved

        Returns
        ---
        none
        """

        # torch.save(self.state_dict(), filename)
        torch.save(self, filename)

    def load_model(self, filename, device):
        """
        Function to load model parameters

        Parameters
        ---
        filename: str
            Location of the file from where the model is to be loaded
        device:
            Device in use - CPU or GPU

        Returns
        ---
        none
        """
        # current_model_dict = self.state_dict()
        # loaded_state_dict = torch.load(filename, map_location=device)
        # new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
        #                   zip(current_model_dict.keys(), loaded_state_dict.values())}
        # self.load_state_dict(self.state_dict(), strict=False)
        # map_location is required to ensure that a model that is trained on GPU can be run even on CPU
        # self.load_state_dict(torch.load(filename, map_location=device), strict=False)# --> was giving an error for
        # load state dict
        torch.load(filename)
