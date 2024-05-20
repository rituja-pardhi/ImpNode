import numpy as np
from livelossplot import PlotLosses
from collections import deque
from livelossplot.outputs import MatplotlibPlot
from torch.utils.tensorboard import SummaryWriter
import time
import csv
import copy


def train_dqn(env, agent, results_base_path, num_train_eps, num_mem_fill_eps, n_step, batch_size,
              update_frequency, train_data_path=None, val_data_path=None, val_step=None, writer=None):
    current_avg_score = 0
    val_score = 0
    liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath='{}/train_plot.png'.format(results_base_path))])
    logs = {}
    last_100_rewards = deque([], maxlen=100)

    best_score = -np.inf

    start_time = time.time()
    for ep_cnt in range(num_train_eps):
        env.data_path = train_data_path
        state_history, action_history, reward_history, mask_history = [], [], [], []
        done = False
        state, info = env.reset(ep_cnt)

        ep_score = 0
        while not done:
            mask = info['node_action_mask']
            original_graph = info['original_graph']

            mask2 = copy.deepcopy(mask)
            action = agent.select_action(state, mask, original_graph)

            next_state, reward, done, truncated, info = env.step(action)

            mask_history.append(mask2)
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            if len(state_history) >= n_step:
                n_step_s_masks = mask_history[-n_step]
                n_step_states = state_history[-n_step]
                n_step_actions = action_history[-n_step]
                n_step_rewards = reward_history[-n_step:]
                n_step_ns_masks = mask_history[-1]
                n_step_ns_masks[action] = 0
                # Calculate n-step return

                n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                agent.memory.store(
                    original_graph=info['original_graph'],
                    s_mask=n_step_s_masks,
                    ns_mask=n_step_ns_masks,
                    state=n_step_states,
                    action=n_step_actions,
                    next_state=next_state,
                    reward=n_step_return,
                    done=done
                )
            state = next_state
            ep_score += reward
        last_100_rewards.append(ep_score)
        current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
        writer.add_scalar('current_avg_score', current_avg_score, ep_cnt)
        logs['train avg score'] = current_avg_score
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()
        if ep_cnt < num_mem_fill_eps:
            continue

        loss = agent.learn(batch_size)
        logs['loss'] = loss.detach().numpy()
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()

        agent.update_epsilon()

        if ep_cnt % val_step == 0:
            env.data_path = val_data_path
            val_score_history = []
            for ep in range(100):
                ep_score = 0
                done = False
                state, info = env.reset(ep)
                while not done:
                    mask = info['node_action_mask']
                    original_graph = info['original_graph']
                    action = agent.select_action(state, mask, original_graph)

                    next_state, reward, done, truncated, _ = env.step(action)
                    ep_score += reward
                    state = next_state

                val_score_history.append(ep_score)

            val_score = np.average(val_score_history)

            logs['val avg score'] = val_score

            if val_score >= best_score:
                agent.save_model('{}/model.pt'.format(results_base_path), ep_cnt)
                best_score = val_score
        writer.add_scalar('val_score', val_score, ep_cnt)

        # update the plots in real-time
        #
        liveloss.update(logs)
        liveloss.send()
    end_time = time.time()
    writer.add_scalar('Total_time', end_time - start_time)


def train_dqn_vm(env, agent, results_base_path, num_train_eps, num_mem_fill_eps, n_step, batch_size,
              update_frequency, train_data_path=None, val_data_path=None, val_step=None, writer=None):
    current_avg_score = 0
    val_score = 0
    logs = {}
    last_100_rewards = deque([], maxlen=100)

    best_score = -np.inf

    start_time = time.time()
    for ep_cnt in range(num_train_eps):
        env.data_path = train_data_path
        state_history, action_history, reward_history, mask_history = [], [], [], []
        done = False
        state, info = env.reset(ep_cnt)

        ep_score = 0
        while not done:
            mask = info['node_action_mask']
            original_graph = info['original_graph']

            mask2 = copy.deepcopy(mask)
            action = agent.select_action(state, mask, original_graph)

            next_state, reward, done, truncated, info = env.step(action)

            mask_history.append(mask2)
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            if len(state_history) >= n_step:
                n_step_s_masks = mask_history[-n_step]
                n_step_states = state_history[-n_step]
                n_step_actions = action_history[-n_step]
                n_step_rewards = reward_history[-n_step:]
                n_step_ns_masks = mask_history[-1]
                n_step_ns_masks[action] = 0
                # Calculate n-step return

                n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                agent.memory.store(
                    original_graph=info['original_graph'],
                    s_mask=n_step_s_masks,
                    ns_mask=n_step_ns_masks,
                    state=n_step_states,
                    action=n_step_actions,
                    next_state=next_state,
                    reward=n_step_return,
                    done=done
                )
            state = next_state
            ep_score += reward
        last_100_rewards.append(ep_score)
        current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
        writer.add_scalar('current_avg_score', current_avg_score, ep_cnt)
        logs['train avg score'] = current_avg_score
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()
        if ep_cnt < num_mem_fill_eps:
            continue

        loss = agent.learn(batch_size)
        logs['loss'] = loss.detach().numpy()
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()

        agent.update_epsilon()

        if ep_cnt % val_step == 0:
            env.data_path = val_data_path
            val_score_history = []
            for ep in range(100):
                ep_score = 0
                done = False
                state, info = env.reset(ep)
                while not done:
                    mask = info['node_action_mask']
                    original_graph = info['original_graph']
                    action = agent.select_action(state, mask, original_graph)

                    next_state, reward, done, truncated, _ = env.step(action)
                    ep_score += reward
                    state = next_state

                val_score_history.append(ep_score)

            val_score = np.average(val_score_history)

            logs['val avg score'] = val_score

            if val_score >= best_score:
                agent.save_model('{}/model.pt'.format(results_base_path), ep_cnt)
                best_score = val_score
        writer.add_scalar('val_score', val_score, ep_cnt)

    end_time = time.time()
    writer.add_scalar('Total_time', end_time - start_time)


def fill_memory(env, agent, num_mem_fill_eps, n_step):
    for ep_cnt in range(num_mem_fill_eps):

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
                    original_graph=info['original_graph'],
                    mask=info['node_action_mask'],
                    state=n_step_states,
                    action=n_step_actions,
                    next_state=next_state,
                    reward=n_step_return,
                    done=done
                )
            state = next_state
            ep_score += reward

def train_dqn_new(env, agent, results_base_path, num_train_eps, num_mem_fill_eps, n_step, batch_size,
              update_frequency, train_data_path=None, val_data_path=None, val_step=None, writer=None):
    current_avg_score = 0
    val_score = 0
    liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath='{}/train_plot.png'.format(results_base_path))])
    logs = {}
    last_100_rewards = deque([], maxlen=100)

    best_score = -np.inf

    start_time = time.time()
    for ep_cnt in range(num_train_eps):
        env.data_path = train_data_path
        state_history, action_history, reward_history, mask_history = [], [], [], []
        done = False
        state, info = env.reset(ep_cnt)
        if ep_cnt % 10==0:
            for i in range(10):
                ep_score = 0
                while not done:
                    mask = info['node_action_mask']
                    original_graph = info['original_graph']

                    mask2 = copy.deepcopy(mask)
                    action = agent.select_action(state, mask, original_graph)

                    next_state, reward, done, truncated, info = env.step(action)

                    mask_history.append(mask2)
                    state_history.append(state)
                    action_history.append(action)
                    reward_history.append(reward)

                    if len(state_history) >= n_step:
                        n_step_s_masks = mask_history[-n_step]
                        n_step_states = state_history[-n_step]
                        n_step_actions = action_history[-n_step]
                        n_step_rewards = reward_history[-n_step:]
                        n_step_ns_masks = mask_history[-1]
                        n_step_ns_masks[action] = 0
                        # Calculate n-step return

                        n_step_return = sum(reward * (agent.discount ** i) for i, reward in enumerate(n_step_rewards))
                        agent.memory.store(
                            original_graph=info['original_graph'],
                            s_mask=n_step_s_masks,
                            ns_mask=n_step_ns_masks,
                            state=n_step_states,
                            action=n_step_actions,
                            next_state=next_state,
                            reward=n_step_return,
                            done=done
                        )
                    state = next_state
                    ep_score += reward
                    last_100_rewards.append(ep_score)
        current_avg_score = np.mean(last_100_rewards)  # get average of last 100 scores
        writer.add_scalar('current_avg_score', current_avg_score, ep_cnt)
        logs['train avg score'] = current_avg_score
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()
        if ep_cnt < num_mem_fill_eps:
            continue

        loss = agent.learn(batch_size)
        logs['loss'] = loss.detach().numpy()
        if ep_cnt % update_frequency == 0:
            agent.update_target_net()

        agent.update_epsilon()

        if ep_cnt % val_step == 0:
            env.data_path = val_data_path
            val_score_history = []
            for ep in range(100):
                ep_score = 0
                done = False
                state, info = env.reset(ep)
                while not done:
                    mask = info['node_action_mask']
                    original_graph = info['original_graph']
                    action = agent.select_action(state, mask, original_graph)

                    next_state, reward, done, truncated, _ = env.step(action)
                    ep_score += reward
                    state = next_state

                val_score_history.append(ep_score)

            val_score = np.average(val_score_history)

            logs['val avg score'] = val_score

            if val_score >= best_score:
                agent.save_model('{}/model.pt'.format(results_base_path), ep_cnt)
                best_score = val_score
        writer.add_scalar('val_score', val_score, ep_cnt)

        # update the plots in real-time
        #
        liveloss.update(logs)
        liveloss.send()
    end_time = time.time()
    writer.add_scalar('Total_time', end_time - start_time)


