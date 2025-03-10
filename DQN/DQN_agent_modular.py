"""
Script that contains details how the DQN agent learns, updates the target network, selects actions and save/loads the model
"""
import time

import networkx as nx
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import get_laplacian

from DQN.model import DQNNet
from DQN.replay_memory import ReplayMemory
from DQN import virtual_node
import torch.optim as optim
from collections import defaultdict

class DQNAgent:
    """
    Class that defines the functions required for training the DQN agent
    """

    def __init__(self, device, alpha, gnn_depth, state_size, hidden_size1, hidden_size2, action_size,
                 discount, eps_max, eps_min, eps_step, memory_capacity, lr, mode, unfrozen_layers=None,
                 dropout1=None, dropout2=None, l2_reg=None):

        self.device = device
        self.alpha = alpha
        self.unfrozen_layers = unfrozen_layers

        # for epsilon-greedy exploration strategy
        self.epsilon_min = eps_min
        self.epsilon_step = eps_step
        self.epsilon = eps_max
        self.epsilon_max = eps_max
        self.gnn_depth = gnn_depth

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount
        self.mode = mode

        # size of the state vectors and number of possible actions
        self.state_size = state_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.action_size = action_size

        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.l2_reg = l2_reg

        # instances of the network for current policy and its target
        self.policy_net = DQNNet(self.gnn_depth, self.state_size, self.hidden_size1, self.hidden_size2,
                                 lr).to(self.device)
        self.target_net = DQNNet(self.gnn_depth, self.state_size, self.hidden_size1, self.hidden_size2,
                                 lr,).to(self.device)

        if self.mode == "finetune":
            for name, child in self.policy_net.named_children():
                if name in self.unfrozen_layers:
                    print(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    print(name + ' is frozen')
                    for param in child.parameters():
                        param.requires_grad = False

        self.target_net.eval()  # since no learning is performed on the target net

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr)#, weight_decay=self.l2_reg)

        if self.mode == 'test':
            self.policy_net.eval()

        # instance of the replay buffer
        self.memory = ReplayMemory(capacity=int(memory_capacity))

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

    def select_action(self, state, mask, original_graph, n=1):
        """
        Uses epsilon-greedy exploration such that, if the randomly generated number is less than epsilon then the agent performs random action, else the agent executes the action suggested by the policy Q-network

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent

        mask: vector of valid nodes (not already removed)
        Returns
        ---
        none
        """

        if random.random() <= self.epsilon:  # amount of exploration reduces with the epsilon value
            valid_actions = np.nonzero(mask)[0]
            a = int(np.random.choice(valid_actions, 1))
            return a

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():

            aux_input = self.calculate_aux_input_one(original_graph,mask,state)
            batch_of_state = self.preprocess_graphs([state]).to(self.device)
            action = self.policy_net.forward(batch_of_state, aux_input).squeeze(1)
            action = action.to(self.device)
            mask = torch.tensor(mask).to(self.device)
            indexes = [(mask == 0).nonzero().squeeze().to(self.device)]
            infinites = (torch.ones(len(indexes)) * float('-inf')).to(self.device)

            action[indexes] = infinites

            if self.mode == "test_multiple":
                a = torch.topk(action, n).indices
            else:
                a = int(torch.topk(action, 1).indices)

        return a  # since actions are discrete, return index that has highest Q

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
            print('memory less than batch size')
            return

        original_graphs, s_masks, ns_masks, states, actions, next_states, rewards, dones = self.memory.sample(int(batchsize), self.device)

        aux_input_states = self.calculate_aux_input(original_graphs, s_masks, states)
        aux_input_next_states = self.calculate_aux_input(original_graphs, ns_masks, next_states)
        # save graphs without virtual node for graph reconstruction loss
        pyg_states_no_vir = [torch_geometric.utils.from_networkx(graph) for graph in states]
        batch_states_no_vir = torch_geometric.data.Batch.from_data_list(pyg_states_no_vir)

        batch_states = self.preprocess_graphs(states)
        adapted_batch_index = torch.cat(
            [batch_states.batch[batch_states.batch == i][:-1] for i in range(batch_states.num_graphs)], dim=0)

        # all q values from policy network and then get q values of the actions that were taken (as in memory)
        # actions vector has to be explicitly reshaped to nx1-vector
        all_q_values_policy, embeddings = self.policy_net.forward(batch_states.to(self.device), aux_input_states, embedding=True)

        all_q_values_policy = all_q_values_policy.squeeze()
        q_values_policy = self.batch_gather(all_q_values_policy, batch_index=adapted_batch_index, dim=1,
                                            gather_index=actions.type(torch.int64).reshape(-1, 1)).squeeze()

        batch_next_states = self.preprocess_graphs(next_states).to(self.device)
        with torch.no_grad():
            q_target = self.target_net.forward(batch_next_states, aux_input_next_states)

        values, argmax_indices = self.batch_max(q_target, adapted_batch_index)
        target_max = values.squeeze()

        td_target = rewards + self.discount * target_max * ~dones

        # calculate the loss as the mean-squared error of td_target and q_values_policy
        self.optimizer.zero_grad()
        loss = F.mse_loss(td_target, q_values_policy, reduction='sum')/batchsize

        if self.alpha:
            # graph reconstruction loss
            loss_recons = self.graph_recon_loss(batch_states_no_vir, embeddings)
            loss += self.alpha * loss_recons

        loss.backward()
        self.optimizer.step()
        return loss.to('cpu')

    def calculate_aux_input(self, original_graphs, masks, states):

        aux_input = torch.Tensor()
        aux_ip = torch.Tensor()
        counter = 0
        twohop_number = 0
        node_twohop_counter = defaultdict(int)
        for i in range(len(original_graphs)):
            aux = []
            num_nodes_removed = np.count_nonzero(masks[i] == 0)
            aux_1 = num_nodes_removed/len(original_graphs[i].nodes)
            c = np.where(masks[i] == 0)[0]

            for edge in original_graphs[i].edges:
                if edge[0] in c or edge[1] in c:
                    counter += 1
                else:
                    if edge[0] in node_twohop_counter:
                        twohop_number += node_twohop_counter[edge[0]]
                        node_twohop_counter[edge[0]] += 1
                    else:
                        node_twohop_counter[edge[0]] = 1

                    if edge[1] in node_twohop_counter:
                        twohop_number += node_twohop_counter[edge[1]]
                        node_twohop_counter[edge[1]] += 1
                    else:
                        node_twohop_counter[edge[1]] = 1
            aux_2 = counter/len(original_graphs[i].edges)
            aux_3 = twohop_number/(len(original_graphs[i].nodes)*len(original_graphs[i].nodes))
            aux_4 = 1

            aux.append([aux_1, aux_2, aux_3, aux_4])
            aux1 = torch.tensor(aux)

            mask = torch.tensor(masks[i])
            mask = mask.unsqueeze(1)

            aux_ip = mask * aux1

            aux_input = torch.cat([aux_input,aux_ip],dim=0)
        return aux_input.to(self.device)
    def calculate_aux_input_one(self, original_graph, mask, state):
        aux = []
        aux_input = torch.Tensor()
        aux_ip = torch.Tensor()

        counter = 0
        twohop_number = 0
        node_twohop_counter = defaultdict(int)

        num_nodes_removed = np.count_nonzero(mask == 0)
        aux_1 = num_nodes_removed/len(original_graph.nodes)
        c = np.where(mask == 0)[0]
        #c = [(masks[i] == 0).nonzero()]

        for edge in original_graph.edges:
            if edge[0] in c or edge[1] in c:
                counter += 1
            else:
                if edge[0] in node_twohop_counter:
                    twohop_number += node_twohop_counter[edge[0]]
                    node_twohop_counter[edge[0]] += 1
                else:
                    node_twohop_counter[edge[0]] = 1

                if edge[1] in node_twohop_counter:
                    twohop_number += node_twohop_counter[edge[1]]
                    node_twohop_counter[edge[1]] += 1
                else:
                    node_twohop_counter[edge[1]] = 1
        aux_2 = counter/len(original_graph.edges)
        aux_3 = twohop_number/(len(original_graph.nodes)*len(original_graph.nodes))
        aux_4 = 1

        aux.append([aux_1, aux_2, aux_3, aux_4])
        aux1 = torch.tensor(aux)

        mask = torch.tensor(mask)
        mask = mask.unsqueeze(1)
        aux_ip = mask * aux1
        aux_input = torch.cat([aux_input, aux_ip], dim=0)
        return aux_input.to(self.device)

    def preprocess_graphs(self, graphs, virtual=False):
        """
        add virtual node and convert to batch
        """
        pyg_state = [torch_geometric.utils.from_networkx(graph, group_node_attrs='all') for graph in graphs]
        transform = virtual_node.VirtualNode()
        data = [transform.forward(graph) for graph in pyg_state]
        batch_of_states = torch_geometric.data.Batch.from_data_list(data)

        if virtual:
            return pyg_state, batch_of_states
        else:
            return batch_of_states

    def graph_recon_loss(self, graph, embedding):
        """
        Function to calculate graph reconstruction loss
        """

        embed = [embedding[graph.batch == i] for i in range(graph.batch_size)]

        laplacians = [torch.sparse_coo_tensor(lap[0], lap[1], (n, n)).to_dense() for lap, n in
                      zip([get_laplacian(graph[i].edge_index) for i in range(graph.batch_size)],
                          [graph[i].num_nodes for i in range(graph.batch_size)])]

        loss_vals = [torch.trace(torch.matmul(torch.transpose(e.to(self.device), 0, 1),
                                              torch.matmul(l.to(self.device), e.to(self.device)))).to(self.device) for
                     l, e in
                     zip(laplacians, embed)]
        loss = sum(loss_vals) / graph.edge_index.size(1)
        return loss.to(self.device)

    def save_model(self, filename, ep_cnt):
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

        self.policy_net.save_model(filename, ep_cnt)

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
        dense_batch, mask = torch_geometric.utils.to_dense_batch(ip.to(self.device), batch_index.to(self.device),
                                                                 fill_value=float("-inf"))
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
        dense_batch, mask = torch_geometric.utils.to_dense_batch(ip.to(self.device), batch_index.to(self.device))
        return torch.gather(dense_batch, dim=dim, index=gather_index)
