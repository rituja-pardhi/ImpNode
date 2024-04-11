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

    def __init__(self, depth, input_size, hidden_size1, hidden_size2, lr=1e-3):
        #super(DQNNet, self).__init__()

        super().__init__()
        self.depth = depth
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, hidden_size1)
        self.linear4 = nn.Linear(2 * hidden_size1, hidden_size1)
        self.sum_agg = SumAgg()
        self.linear5 = nn.Linear(hidden_size1*hidden_size1, hidden_size1)

        self.dense1 = nn.Linear(hidden_size1, hidden_size2)
        self.dense2 = nn.Linear(hidden_size2, 1)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)

    def forward(self, data, embedding=False):
        x, edge_index = data.x.to(torch.float32), data.edge_index

        # x, edge_index, edge_attr = data.x.to(torch.float32), data.edge_index, data.edge_attr.to(torch.float32)

        x = F.relu(self.linear1(x))
        x = x / x.norm(dim=-1, keepdim=True)

        for _ in range(self.depth):
            neighbor_messages = self.sum_agg(x, edge_index)

            x = torch.cat([self.linear3(neighbor_messages), self.linear2(x)], dim=-1)
            x = F.relu(self.linear4(x))
            x = x / x.norm(dim=-1, keepdim=True)

        # x = torch.cat([torch.cat(
        #     (x[data.batch == i][:-1], x[data.batch == i][-1].repeat(len(x[data.batch == i]) - 1, 1)), dim=1) for i
        #     in range(data.num_graphs)])
        embed = torch.cat([x[data.batch == i][:-1] for i in range(data.num_graphs)])

        x = torch.cat([torch.matmul(x[data.batch == i][:-1].unsqueeze(2),x[data.batch == i][-1].unsqueeze(1).T.unsqueeze(0)) for i in range(data.num_graphs)])
        x = torch.flatten(x, start_dim=1)
        x = self.linear5(x)

        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        if embedding:
            return x, embed

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

        torch.save(self.state_dict(), filename)
        # torch.save(self, filename) -->cannot cope with changing parameters

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
        current_model_dict = self.state_dict()
        loaded_state_dict = torch.load(filename, map_location=device)
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        self.load_state_dict(new_state_dict, strict=False)
        # map_location is required to ensure that a model that is trained on GPU can be run even on CPU
        # self.load_state_dict(torch.load(filename, map_location=device), strict=False)# --> was giving an error for
        # load state dict
        # torch.load(filename) -->cannot cope with changing parameters
