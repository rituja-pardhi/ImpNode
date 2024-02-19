import os
import gym
import numpy as np
import pickle
from livelossplot import PlotLosses
from collections import deque
import networkx as nx

import torch
import DQN_agent
from envs.GraphEnv.impnode import ImpnodeEnv


# to initialize the replay buffer with some random interactions

def fill_memory(env, agent):
    for _ in range(NUM_MEM_FILL_EPS):
        done = False
        state, info = env.reset()

        while not done:
            action = env.action_space.sample(mask=info['node_action_mask'])  # samples random action
            next_state, reward, done, truncated, info = env.step(action)
            agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)


# trains the agent and plots the associated moving average rewards and epsilon values in real-time

def train_loop(env, agent, results_basepath):
    liveloss = PlotLosses()
    logs = {}

    last_100_rewards = deque([], maxlen=100)

    reward_history = []
    epsilon_history = []

    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(NUM_TRAIN_EPS):

        logs['train epsilon'] = agent.epsilon  # to plot current epsilon value

        done = False
        state, info = env.reset()
        ep_score = 0
        while not done:
            mask = info['node_action_mask']
            action = agent.select_action(state, mask)

            next_state, reward, done, truncated, info = env.step(action)

            if step_cnt >= 5:
                agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)
                agent.learn(BATCHSIZE)

                if step_cnt % UPDATE_FREQUENCY == 0:
                    agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        agent.update_epsilon()

        last_100_rewards.append(ep_score)
        current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
        logs['train avg score'] = current_avg_score

        reward_history.append(ep_score)
        epsilon_history.append(logs['train epsilon'])

        if current_avg_score >= best_score:
            agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score

        # update the plots in real-time
        liveloss.update(logs)
        liveloss.send()

    # store the reward and epsilon history that was tracked while running locally

    with open('{}/train_reward_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(epsilon_history, f)




if __name__ == "__main__":
    import sys

    # sys.path.append('C:/Users/rituja.pardhi/Thesis/ma-rituja-pardhi/envs/GraphEnv')
    print(sys.path)
    print("Current working directory:", os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # variables for training the agent

    NUM_TRAIN_EPS = 100000  # 1000 number training episodes to run
    NUM_MEM_FILL_EPS = 1  # 10 number of episodes to run to initialize the memory

    DISCOUNT = 0.99  # gamma used for computing return

    BATCHSIZE = 64  # number of transitions to sample from replay buffer for each learn step
    MEMORY_CAPACITY = 64  # size of the memory buffer
    UPDATE_FREQUENCY = 10  # number of interactions after which the target buffer is updated

    EPS_MAX = 1.0  # initial epsilon value
    EPS_MIN = 0.05  # final epsilon value
    EPS_STEP = 10  # amount by which epsilon is decayed at each episode

    LR = 0.01  # learning rate for the network

    # create folder for storing the model and other files
    results_basepath_train = "results/new_30-50_traineps{}_epsmax{}_epsmin{}_epsstep{}_batchsize{}_treps{}_memeps{}_memcap{}_gseed{}".format(
        NUM_TRAIN_EPS,
        EPS_MAX,
        EPS_MIN,
        EPS_STEP,
        BATCHSIZE,
        NUM_TRAIN_EPS,
        NUM_MEM_FILL_EPS,
        MEMORY_CAPACITY,
        False)
    os.makedirs(results_basepath_train, exist_ok=True)

    seed = None
    env_train = ImpnodeEnv(ba_edges=4, max_removed_nodes=3, seed=seed, render_option=False, data=False, train_mode=True)

    # create the dqn_agent
    dqn_agent_train = DQN_agent.DQNAgent(device,
                                         # env_train.observation_space.shape[0],
                                         5,
                                         env_train.action_space.n,
                                         discount=DISCOUNT,
                                         eps_max=EPS_MAX,
                                         eps_min=EPS_MIN,
                                         eps_step=EPS_STEP,
                                         memory_capacity=MEMORY_CAPACITY,
                                         lr=LR,
                                         train_mode=True)

    # initialise the memory
    fill_memory(env_train, dqn_agent_train)

    # train the agent
    train_loop(env_train, dqn_agent_train, results_basepath_train)
