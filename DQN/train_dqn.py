import os
import numpy as np
import pickle
from livelossplot import PlotLosses
from collections import deque

from pytorchtools import EarlyStopping
import torch
import time
import csv


# to initialize the replay buffer with some random interactions

def fill_memory(env, agent, num_mem_fill_eps, n_step):
    for _ in range(num_mem_fill_eps):

        state_history, action_history, reward_history = [], [], []
        done = False
        state, info = env.reset()

        while not done:
            action = env.action_space.sample(mask=info['node_action_mask'])  # samples random action
            next_state, reward, done, truncated, info = env.step(action)
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            if len(state_history) >= n_step:
                n_step_states = state_history[-n_step]
                n_step_actions = action_history[-n_step]
                n_step_rewards = reward_history[-n_step:]

                # Calculate n-step return
                n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                agent.memory.store(
                    state=n_step_states,
                    action=n_step_actions,
                    next_state=next_state,
                    reward=n_step_return,
                    done=done
                )
            state = next_state


def fill_memory2(env, agent, num_mem_fill_eps, n_step):
    for ep in range(num_mem_fill_eps):

        state_history, action_history, reward_history = [], [], []
        done = False
        state, info = env.reset(ep)

        while not done:
            action = env.action_space.sample(mask=info['node_action_mask'])  # samples random action
            next_state, reward, done, truncated, info = env.step(action)
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            if len(state_history) >= n_step:
                n_step_states = state_history[-n_step]
                n_step_actions = action_history[-n_step]
                n_step_rewards = reward_history[-n_step:]

                # Calculate n-step return
                n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))

                agent.memory.store(
                    state=n_step_states,
                    action=n_step_actions,
                    next_state=next_state,
                    reward=n_step_return,
                    done=done
                )
            state = next_state


def train_dqn(env, agent, results_base_path, num_train_eps, n_step, batch_size, update_frequency):

    early_stopping = EarlyStopping(patience=5, verbose=True)

    liveloss = PlotLosses()
    logs = {}
    last_100_rewards = deque([], maxlen=100)

    step_cnt = 0
    best_score = -np.inf

    with open('{}/info.csv'.format(results_base_path), 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['ep_cnt', 'train_avg_score', 'val_avg_score'])
        for ep_cnt in range(num_train_eps):

            state_history, action_history, reward_history = [], [], []
            done = False
            state, info = env.reset()
            ep_score = 0
            while not done:
                mask = info['node_action_mask']
                action = agent.select_action(state, mask)

                next_state, reward, done, truncated, info = env.step(action)

                state_history.append(state)
                action_history.append(action)
                reward_history.append(reward)

                if len(state_history) >= n_step:
                    n_step_states = state_history[-n_step]
                    n_step_actions = action_history[-n_step]
                    n_step_rewards = reward_history[-n_step:]

                    # Calculate n-step return
                    n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                    agent.memory.store(
                        state=n_step_states,
                        action=n_step_actions,
                        next_state=next_state,
                        reward=n_step_return,
                        done=done
                    )
                    agent.learn(batch_size)

                    if step_cnt % update_frequency == 0:
                        agent.update_target_net()

                    step_cnt += 1

                state = next_state
                ep_score += reward

            agent.update_epsilon()

            last_100_rewards.append(ep_score)
            current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
            logs['train avg score'] = current_avg_score

            if ep_cnt % 300 == 0:
                agent.data = True
                val_score_history = []
                for ep in range(200):
                    ep_score = 0
                    done = False
                    state, info = env.reset(ep)
                    while not done:
                        mask = info['node_action_mask']
                        action = agent.select_action(state, mask)
                        next_state, reward, done, truncated, _ = env.step(action)
                        ep_score += reward
                        state = next_state

                    val_score_history.append(ep_score)
                    agent.data = False
                val_score = np.average(val_score_history)

                early_stopping(val_score, agent)
                logs['val avg score'] = val_score
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if val_score >= best_score:
                    agent.save_model('{}/model.pt'.format(results_base_path))
                    best_score = val_score

            # update the plots in real-time
            liveloss.update(logs)
            liveloss.send()
            csv_writer.writerow([ep_cnt, current_avg_score, val_score])

    # store the reward and epsilon history that was tracked while running locally
    return current_avg_score


def finetune_dqn2(agent, num_train_eps, results_base_path, batch_size):
    for ep_cnt in range(num_train_eps):
        agent.learn(batch_size)
        agent.update_target_net()
        agent.save_model('{}/er_5_4_finetuned_model.pt'.format(results_base_path))


def finetune_trial(env, agent, results_base_path, num_train_eps, n_step, batch_size, update_frequency, finetuned_base_path):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    liveloss = PlotLosses()
    logs = {}
    last_100_rewards = deque([], maxlen=100)

    step_cnt = 0
    best_score = -np.inf

    with open('{}/finetune_info.csv'.format(results_base_path), 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['ep_cnt', 'train_avg_score', 'val_avg_score'])
        for ep_cnt in range(num_train_eps):

            state_history, action_history, reward_history = [], [], []
            done = False
            state, info = env.reset(ep_cnt)
            ep_score = 0
            while not done:
                mask = info['node_action_mask']
                action = agent.select_action(state, mask)

                next_state, reward, done, truncated, info = env.step(action)

                state_history.append(state)
                action_history.append(action)
                reward_history.append(reward)

                if len(state_history) >= n_step:
                    n_step_states = state_history[-n_step]
                    n_step_actions = action_history[-n_step]
                    n_step_rewards = reward_history[-n_step:]

                    # Calculate n-step return
                    n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                    agent.memory.store(
                        state=n_step_states,
                        action=n_step_actions,
                        next_state=next_state,
                        reward=n_step_return,
                        done=done
                    )
                    agent.learn(batch_size)

                    if step_cnt % update_frequency == 0:
                        agent.update_target_net()

                    step_cnt += 1

                state = next_state
                ep_score += reward

            agent.update_epsilon()

            last_100_rewards.append(ep_score)
            current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
            logs['train avg score'] = current_avg_score

            if ep_cnt % 100 == 0:
                agent.data = True
                val_score_history = []
                for ep in range(50):
                    ep_score = 0
                    done = False
                    state, info = env.reset(ep)
                    while not done:
                        mask = info['node_action_mask']
                        action = agent.select_action(state, mask)
                        next_state, reward, done, truncated, _ = env.step(action)
                        ep_score += reward
                        state = next_state

                    val_score_history.append(ep_score)
                    agent.data = False
                val_score = np.average(val_score_history)

                early_stopping(val_score, agent)
                logs['val avg score'] = val_score
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if val_score >= best_score:
                    agent.save_model('{}/{}.pt'.format(results_base_path,finetuned_base_path))
                    best_score = val_score

            # update the plots in real-time
            liveloss.update(logs)
            liveloss.send()
            csv_writer.writerow([ep_cnt, current_avg_score, val_score])

    # store the reward and epsilon history that was tracked while running locally
    return current_avg_score
