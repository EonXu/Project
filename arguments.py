import os
import argparse
import numpy as np


SEED = [19990227, 19991023, 19980123, 19990417]


def get_current_seeds():
    root = './result'
    files = os.listdir(root)
    seeds = []
    for f in files:
        tmps = f.split('_')
        for tmp in tmps:
            if tmp.find('Seed') > -1:
                seeds.append(int(tmp.split('Seed')[1]))
    # print(seeds)
    return seeds


def get_common_args():
    # 创建了一个参数解析器的实例。这个实例用于定义程序能够接受的命令行参数，并解析用户提供的输入
    parser = argparse.ArgumentParser()

    # 环境相关的设置
    parser.add_argument('--env', type=str, default='flight_easy', help='the environment of the experiment')
    parser.add_argument('--map_size', type=int, default=50, help='the size of the grid map')
    parser.add_argument('--target_num', type=int, default=15, help='the num of the search targets')
    parser.add_argument('--target_mode', type=int, default=0, help='targets location mode')
    parser.add_argument('--target_dir', type=str, default='./targets/', help='targets directory')
    parser.add_argument('--agent_mode', type=int, default=0, help='agents location mode')       # bottom line
    parser.add_argument('--n_agents', type=int, default=4, help='the num of agents')
    parser.add_argument('--view_range', type=int, default=7, help='the view range of agent')
    # algorithms args
    parser.add_argument('--alg', type=str, default='d3qn', help='the algorithms for training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use common network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agents')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    #True改为False
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--seed_idx', type=int, default=4, help='the index of the model seed list')
    # train or show model
    parser.add_argument('--show', type=bool, default=False, help='train or show model')
    parser.add_argument('--experiment', type=bool, default=False, help='whether to collect experimental data for certain algorithm')
    args = parser.parse_args()
    return args

# arguments of central_v
def get_reinforce_args(args):
    # network
    args.off_policy = False
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.tau = 0.05
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3
    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1000)
    # the number of the epoch to train the agent
    args.n_epoch = 100000
    # the number of the episodes in one epoch
    args.n_episodes = 1
    # how often to evaluate
    args.evaluate_cycle = 200
    # how often to save the model
    args.save_cycle = 5
    # prevent gradient explosion
    args.grad_norm_clip = 10
    return args
def get_ddqn_args(args):
    args.off_policy = True
    args.rnn_hidden_dim = 64
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.epsilon_anneal_scale = 'epoch'
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)
    # the number of the train step in one epoch
    args.train_steps = 1
    args.lr = 1e-4
    # the number of the epoch to train the agent
    args.n_epoch = 100000
    # how often to evaluate
    args.evaluate_cycle = 200
    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to save the model
    args.save_cycle = 500
    return args

def get_doubledqn_args(args):
    args.off_policy = False
    args.rnn_hidden_dim = 64
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.epsilon_anneal_scale = 'epoch'
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)
    # the number of the train step in one epoch
    args.train_steps = 1
    args.lr = 1e-4
    # the number of the epoch to train the agent
    args.n_epoch = 100000
    # how often to evaluate
    args.evaluate_cycle = 10
    # the number of the episodes in one epoch
    args.n_episodes = 1
    args.target_update_cycle = 50

    # how often to save the model
    args.save_cycle = 10
    return args

def get_dqn_args(args):
    args.off_policy = False
    args.rnn_hidden_dim = 64
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.epsilon_anneal_scale = 'epoch'
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)
    # the number of the train step in one epoch
    args.train_steps = 1
    args.lr = 1e-4
    # the number of the epoch to train the agent
    args.n_epoch = 100000
    # how often to evaluate
    args.evaluate_cycle = 200
    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to save the model
    args.save_cycle = 200
    return args

def get_d3qn_args(args):
    args.off_policy = False
    args.rnn_hidden_dim = 64
    args.seed = 37763296
    # if not args.show and not args.load_model:
    #     if args.seed_idx < len(SEED):
    #         args.seed = SEED[args.seed_idx]
    #     else:
    #         currseeds = get_current_seeds()
    #         while True:
    #             seed = np.random.randint(10000000, 99999999)
    #             if seed not in currseeds:
    #                 args.seed = seed
    #                 break
    args.epsilon_anneal_scale = 'epoch'
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)
    # the number of the train step in one epoch
    args.train_steps = 1
    args.lr = 1e-4
    # the number of the epoch to train the agent
    args.n_epoch = 100000
    # how often to evaluate
    args.evaluate_cycle = 200
    # the number of the episodes in one epoch
    args.n_episodes = 1

    args.target_update_cycle = 50

    args.tau = 0.05
    # how often to save the model
    args.save_cycle = 10
    return args

#用于获取传统任务的参数，例如使用强化学习的一些基本参数。其中，将off_policy设置为False，而其他参数都被设置为零。
def get_traditional_args(args):
    args.off_policy = False
    args.epsilon = 0
    args.anneal_epsilon = 0
    args.min_epsilon = 0
    args.n_epoch = 100000
    args.epsilon_anneal_scale = 'epoch'
    # how often to evaluate
    args.evaluate_cycle = 10
    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to save the model
    args.save_cycle = 200
    args.seed = 0
    return args

# args of flight search env
def get_flight_easy_args(args):
    # flight info
    args.agent_velocity = 1
    args.time_limit = 200
    args.turn_limit = np.pi/4           # rad/s
    args.flight_height = 8000
    args.safe_dist = 1
    args.detect_prob = 0.9
    args.wrong_alarm_prob = 0.1
    args.force_dist = 3
    args.search_env = True
    # env
    args.conv = False
    return args

#获取飞行搜索任务场景的参数。它设置了一系列与任务相关的参数，例如飞行速度、时间限制、高度等等。
#同时，它在搜索环境中进行。还在环境中使用卷积。
def get_flight_args(args):
    # flight info
    args.agent_velocity = 1
    args.time_limit = 200
    args.turn_limit = np.pi/4           # rad/s
    args.flight_height = 8000
    args.safe_dist = 1
    args.detect_prob = 0.9
    args.wrong_alarm_prob = 0.1
    args.force_dist = 3
    args.search_env = True

    # env
    args.conv = True
    args.dim_1 = 4
    args.kernel_size_1 = 4
    args.stride_1 = 2
    # args.padding = 2
    args.dim_2 = 1
    args.kernel_size_2 = 3
    args.stride_2 = 1
    args.padding_2 = 1

    args.conv_out_dim = 16

    return args
