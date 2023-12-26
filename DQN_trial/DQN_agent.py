"""
Script that contains details how the DQN agent learns, updates the target network, selects actions and save/loads the model
"""

import networkx as nx
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter
from torch_geometric.utils import to_dense_adj

from model import DQNNet
from replay_memory import ReplayMemory


class DQNAgent:
    """
    Class that defines the functions required for training the DQN agent
    """

    def __init__(self, device, state_size, action_size,
                 discount=0.99,
                 eps_max=1.0,
                 eps_min=0.05,
                 eps_step=10000,
                 memory_capacity=5000,
                 lr=1e-3,
                 train_mode=True):

        self.device = device

        # for epsilon-greedy exploration strategy
        self.epsilon_max = eps_max
        self.epsilon_min = eps_min
        self.epsilon_step = eps_step
        self.epsilon = 1

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount

        # size of the state vectors and number of possible actions
        self.state_size = state_size
        self.action_size = action_size

        # instances of the network for current policy and its target
        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval()  # since no learning is performed on the target net

        if not train_mode:
            self.policy_net.eval()

        # instance of the replay buffer
        self.memory = ReplayMemory(capacity=memory_capacity)

    def update_target_net(self):
        """
        Function to copy the weights of the current policy net into the (frozen) target net

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Function for reducing the epsilon value (used for epsilon-greedy exploration with annealing)

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.epsilon -= (self.epsilon_max - self.epsilon_min) / self.epsilon_step
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state):
        """
        Uses epsilon-greedy exploration such that, if the randomly generated number is less than epsilon then the agent performs random action, else the agent executes the action suggested by the policy Q-network
        """
        """
        Function to return the appropriate action for the given state.
        During training, returns a randomly sampled action or a greedy action (predicted by the policy network), based on the epsilon value.
        During testing, returns action predicted by the policy network

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent

        Returns
        ---
        none
        """

        if random.random() <= self.epsilon:  # amount of exploration reduces with the epsilon value
            return random.randrange(self.action_size)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            new_state = state.to_directed()
            new_node = len(new_state)
            new_state.add_node(new_node)
            # Add directed edges from the new node to all existing nodes
            for node in state.nodes:
                new_state.add_edge(new_node, node)
            nx.set_node_attributes(new_state, {new_node: np.ones(5, dtype=int)}, name='features')
            pyg_state = torch_geometric.utils.from_networkx(new_state)
            action = self.policy_net.forward(pyg_state)
        return torch.argmax(action).item()  # since actions are discrete, return index that has highest Q

    def learn(self, batchsize):
        """
        Function to perform the updates on the neural network that runs the DQN algorithm.

        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from

        Returns
        ---
        none
        """

        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)
        trial_pyg_states = [torch_geometric.utils.from_networkx(new_state) for new_state in states]

        new_states = []
        # add virtual node, it's edges and it's features for states
        for state in states:
            new_state = state.to_directed()
            new_node = len(new_state)
            new_state.add_node(new_node)
            # Add directed edges from the new node to all existing nodes
            for node in state.nodes:
                new_state.add_edge(new_node, node)
            nx.set_node_attributes(new_state, {new_node: np.ones(5, dtype=int)}, name='features')
            new_states.append(new_state)

        # nx to pyg graph conversion for states
        pyg_states = [torch_geometric.utils.from_networkx(new_state) for new_state in new_states]
        batch_of_states = torch_geometric.data.Batch.from_data_list(pyg_states)

        adapted_batch_index = torch.cat(
            [batch_of_states.batch[batch_of_states.batch == i][:-1] for i in range(batch_of_states.num_graphs)], dim=0)

        # all q values from policy network and then get q values of the actions that were taken (as in memory)
        # actions vector has to be explicitly reshaped to nx1-vector
        all_q_values_policy = self.policy_net.forward(batch_of_states).squeeze()
        q_values_policy = self.batch_gather(all_q_values_policy, batch_index=adapted_batch_index, dim=1,
                                            gather_index=actions.type(torch.int64).reshape(-1, 1)).squeeze()

        # add virtual node, it's edges and it's features for next_states
        new_next_states = []
        # add virtual node, it's edges and it's features for states
        for next_state in next_states:
            new_next_state = next_state.to_directed()
            new_node = len(new_next_state)
            new_next_state.add_node(new_node)
            # Add directed edges from the new node to all existing nodes
            for node in next_state.nodes:
                new_next_state.add_edge(new_node, node)
            nx.set_node_attributes(new_next_state, {new_node: np.ones(5, dtype=int)}, name='features')
            new_next_states.append(new_next_state)

        pyg_next_states = [torch_geometric.utils.from_networkx(state) for state in new_next_states]
        batch_of_next_states = torch_geometric.data.Batch.from_data_list(pyg_next_states)

        q_target = self.target_net.forward(batch_of_next_states)
        values, argmax_indices = self.batch_max(q_target, adapted_batch_index)
        target_max = values.squeeze()
        td_target = rewards + self.discount * target_max * ~dones

        #
        graph = nx.barabasi_albert_graph(8, 2, seed=1)
        nx.set_node_attributes(graph, np.ones(5, dtype=int), 'features')
        graph = torch_geometric.utils.from_networkx(graph)
        embedding = self.policy_net.forward(graph, embedding=True)
        pairwise_distances = torch.cdist(embedding, embedding)

        adjacency_matrix = to_dense_adj(graph.edge_index).squeeze(0)

        loss_recon = torch.sum(torch.mul(adjacency_matrix, pairwise_distances))/2
        print(loss_recon)



        # calculate the loss as the mean-squared error of td_target and q_values_policy
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(td_target, q_values_policy) + 0.01*loss_recon
        loss.backward()
        self.policy_net.optimizer.step()

    def save_model(self, filename):
        """
        Function to save the policy network

        Parameters
        ---
        filename: str
            Location of the file where the model is to be saved

        Returns
        ---
        none
        """

        self.policy_net.save_model(filename)

    def load_model(self, filename):
        """
        Function to load model parameters

        Parameters
        ---
        filename: str
            Location of the file from where the model is to be loaded

        Returns
        ---
        none
        """

        self.policy_net.load_model(filename=filename, device=self.device)

    def batch_max(self, ip, batch_index, dim=1):
        dense_batch, mask = torch_geometric.utils.to_dense_batch(ip, batch_index, fill_value=float("-inf"))
        max_values, argmax_indices = torch.max(dense_batch, dim=dim)
        return max_values, argmax_indices

    def batch_gather(self, ip, batch_index, dim, gather_index):
        """
        Gathers values from the input tensor along a specified dimension based on batch and gather indices.

        Parameters:
        - input (Tensor): The source tensor from which to gather values.
        - batch_index (LongTensor): The indices that indicate which element of the input belongs to which graph.
        - dim (int): The dimension along which to index.
        - gather_index (LongTensor): The indices of elements to gather. It should have the same number of dimensions as input,
          except for the specified dimension dim where the indices are taken.

        Returns:
        - Tensor: The gathered values.


        """
        dense_batch, mask = torch_geometric.utils.to_dense_batch(ip, batch_index)
        return torch.gather(dense_batch, dim=dim, index=gather_index)
