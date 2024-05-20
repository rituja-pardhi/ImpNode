import os
import numpy as np
import random
import torch
import time
from tqdm import tqdm
from pathlib import Path
from DQN.train_dqn_vm import train_dqn_vm
import mlflow
from envs.GraphEnv.impnode import ImpnodeEnv
from torch.utils.tensorboard import SummaryWriter
from DQN import DQN_agent_modular
from DQN.test_and_compare import test_loop


def train(results_base_path_train, g_type, train_data_path, val_data_path, depth, hidden1, hidden2, lr):
    seed = 412
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with mlflow.start_run(run_name='t-{}'.format(g_type)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        results_base_path_train_new = results_base_path_train + "/{}".format(g_type)

        timestamp = time.strftime("%Y%m%d%H%M%S")
        new_directory = f"{results_base_path_train_new}_{timestamp}"
        os.makedirs(new_directory)
        results_base_path_train_new = new_directory
        print(f"Directory already exists. Created a new one: {new_directory}")

        params = {
            'device': device,
            'seed': seed,
            'aux_dim': True,

            'num_train_eps': 20000,
            'num_mem_fill_eps': 100,
            'discount': 0.99,
            'batch_size': 64,
            'mem_capacity': 500000,
            'update_frequency': 500,
            'eps_max': 1.0,
            'eps_min': 0.05,
            'eps_step': 10000,
            'lr': lr,
            'alpha': 0.001,

            'gnn_depth': depth,
            'state_size': 2,
            'hidden_size1': hidden1,
            'hidden_size2': hidden2,
            'action_size': 1,
            'n_step': 5,

            'g_type': g_type,
            'num_nodes': (30, 50),
            'anc': 'dw_nd',
            'max_removed_nodes': None,

            'train_data_path': train_data_path,
            'val_data_path': val_data_path,
            'val_step': 1000,

            'results_base_path_train': results_base_path_train_new
        }
        mlflow.log_params(params)

        writer = SummaryWriter(results_base_path_train + '/tensorboard/{}_{}'.format(g_type, timestamp))

        env_train = ImpnodeEnv(anc=params['anc'],
                               g_type=g_type,
                               num_nodes=params['num_nodes'],
                               mode='train',
                               data_path=params['train_data_path']
                               )

        dqn_agent_train = DQN_agent_modular.DQNAgent(device,
                                                     alpha=params['alpha'],
                                                     gnn_depth=params['gnn_depth'],
                                                     state_size=params['state_size'],
                                                     hidden_size1=params['hidden_size1'],
                                                     hidden_size2=params['hidden_size2'],
                                                     action_size=params['action_size'],
                                                     discount=params['discount'],
                                                     eps_max=params['eps_max'],
                                                     eps_min=params['eps_min'],
                                                     eps_step=params['eps_step'],
                                                     memory_capacity=params['mem_capacity'],
                                                     lr=params['lr'],
                                                     mode='train')

        train_dqn_vm(env_train, dqn_agent_train, results_base_path_train_new, params['num_train_eps'],
                     params['num_mem_fill_eps'], params['n_step'], params['batch_size'], params['update_frequency'],
                     train_data_path=params['train_data_path'], val_data_path=params['val_data_path'],
                     val_step=params['val_step'], writer=writer)

        subdir = params['val_data_path']
        data_path = Path.cwd() / subdir

        env_test_val = ImpnodeEnv(anc='dw_nd', data_path=data_path, mode='test')

        actions, reward_history, ep_score_history = test_loop(env=env_test_val,
                                                                 agent=dqn_agent_train,
                                                                 NUM_TEST_EPS=20)
        mlflow.log_metric('anc_val', np.mean(ep_score_history))



if __name__ == '__main__':

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('VM_GPU_2')

    train_data_types = ['erdos-renyi','watts-strogatz']
    train_data_paths = [None, None]
    val_data_paths = ['data/synthetic_small_dataset/ER_val_30_50','data/synthetic_small_dataset/WS_val_30_50']
    results_base_path_train = 'results/train_baseline_model_vm_gpu_2'

    depths = [3, 4]
    hidden_dim_1 = [32, 64]
    hidden_dim_2 = [32, 64]
    lrs = [0.0001, 0.00001]

    for train_data, train_data_path, val_data_path in zip(train_data_types, train_data_paths, val_data_paths):
        for depth in depths:
            for hidden1 in hidden_dim_1:
                for hidden2 in hidden_dim_2:
                    for lr in lrs:
                        train(results_base_path_train, g_type=train_data, train_data_path=train_data_path,
                              val_data_path=Path.cwd() / val_data_path, depth=depth,
                              hidden1=hidden1, hidden2=hidden2, lr=lr)
                        print('Finished training')
