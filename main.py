import os
import torch
import pynvml#这是一个与NVIDIA管理库（NVML）交互的Python库，用于监控和管理NVIDIA GPU。
from runner import Runner
from env.flight_env_easy import FlightSearchEnvEasy
from env.flight_env import FlightSearchEnv
from env.test_env import TestSearchEnv
from arguments import get_common_args, get_reinforce_args,get_ddqn_args,get_dqn_args,get_flight_easy_args,get_flight_args,get_d3qn_args,get_traditional_args,get_doubledqn_args
print(torch.__version__)
print(torch.cuda.is_available())

#CIRCLE_DICT_1可能用于环境配置，如定义圆形区域的中心、半径和目标数量。TARGETS_FILENAME定义了一个路径，可能用于存储或加载目标位置的文件。
CIRCLE_DICT_1 = {'circle_center': [[10, 32], [12, 15]], 'circle_radius': [5, 7], 'target_num': [7, 8]}
TARGETS_FILENAME = './'


def load_targets(filename):
    # 打开文件以读取目标信息
    f = open(filename, 'r')
    # 读取文件的所有行，去掉第一行（假设第一行是标题）
    lines = f.readlines()[1:]
    # 初始化空列表以存储不同属性的目标信息
    x, y, deter, priority, dx, dy = [], [], [], [], [], []
    # 遍历文件中的每一行
    for line in lines:
        # 将每一行拆分成单词
        line = line.split()
        # 将目标属性的值转换为相应的数据类型并添加到相应的列表中
        x.append(float(line[0]))
        y.append(float(line[1]))
        deter.append(line[2])
        priority.append(int(line[3]))
        dx.append(float(line[4]))
        dy.append(float(line[5]))
    # 创建一个字典，将不同属性的列表存储在相应的键下
    CIRCLE_DICT = {'x': x, 'y': y, 'deter': deter, 'priority': priority, 'dx': dx, 'dy': dy}
    # 返回包含目标信息的字典
    return CIRCLE_DICT


def smart_gpu_allocate():
    # 打印初始化智能GPU分配的消息
    print('Init smart GPU allocation')
    # 初始化NVML（NVIDIA Management Library）
    pynvml.nvmlInit()
    # 获取系统中可用的GPU数量
    gpu_num = pynvml.nvmlDeviceGetCount()
    print('ALL {} gpus list:'.format(gpu_num))
    # 初始化变量，用于跟踪具有最大空闲内存的GPU
    max_idx = 0
    max_free_mem = 0
    # 遍历每个GPU
    for gpu_idx in range(gpu_num):
        # 获取GPU句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        # 获取GPU的名称
        gpu_name = str(pynvml.nvmlDeviceGetName(handle), 'utf-8')
        # 获取GPU的内存信息
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = meminfo.total  # 总内存
        used_mem = meminfo.used    # 已使用内存
        free_mem = meminfo.free    # 空闲内存
        # 打印GPU的名称、已使用内存和总内存信息
        print('{} ({}/{} M)'.format(gpu_name, used_mem // 1024**2, total_mem // 1024**2))
        # 如果当前GPU的空闲内存大于记录的最大空闲内存
        if free_mem > max_free_mem:
            # 更新最大空闲内存和对应的GPU索引
            max_free_mem = free_mem
            max_idx = gpu_idx
    # 打印选择的GPU索引
    print('Using GPU {}'.format(max_idx))
    # 设置CUDA_VISIBLE_DEVICES环境变量，限制PyTorch使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max_idx)
    # 关闭NVML
    pynvml.nvmlShutdown()

def get_experiment_seed(args):
    # 获取指定目录（'./model'）下的所有文件列表
    files = os.listdir('./model')
    # 遍历每个文件
    for f in files:
        # 使用'Seed'分割文件名
        tmps = f.split('Seed')
        # 从文件名中提取环境名称
        env = tmps[0][:-1]
        # 构建与实验参数相关的模型名称
        name = args.alg + '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode, args.target_mode)
        # 检查文件是否与当前环境、算法和参数匹配
        if env == args.env and tmps[1].find(name) > -1:
            # 从文件名中提取种子值并转换为整数
            seed = int(tmps[1].split('_')[0])
            # 返回找到的种子值
            return seed
    # 如果没有找到匹配的文件，返回-1
    return -1


if __name__ == "__main__":
    # 获取通用的命令行参数
    args = get_common_args()
    # 如果是展示模型
    if not args.show:
        # 设置PyTorch的线程数限制
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    # 如果是加载模型或加载预训练的模型
    if args.show or args.load_model:
        # 获取实验的种子值
        seed = get_experiment_seed(args)
        if seed != -1:
            # 如果找到种子值，将其设置为参数中的种子值
            args.seed = seed
    # 加载算法参数
    # if args.alg.find('qmix') > -1 or args.alg.find('vdn') > -1:
    #     args = get_mixer_args(args)
    # elif args.alg.find('dop') > -1:
    #     args = get_dop_args(args)
    if args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    elif args.alg.find('d3qn') > -1:
        args = get_d3qn_args(args)
    elif args.alg.find('dqn') > -1:
        args = get_dqn_args(args)
    elif args.alg.find('ddqn') > -1:
        args = get_ddqn_args(args)
    elif args.alg.find('doubledqn') > -1:
        args = get_doubledqn_args(args)
    elif args.alg.find('random') > -1:
        args = get_traditional_args(args)

    # load environment arguements
    # if args.env == 'search':
    #     args = get_test_search_args(args)
    #     env = SearchEnv(args, circle_dict=CIRCLE_DICT_1)
    if args.env == 'flight':
        args = get_flight_args(args)
        targets_root = './flight_targets.txt'
        CIRCLE_DICT_2 = load_targets(targets_root)
        env = FlightSearchEnv(args, circle_dict=CIRCLE_DICT_2)
    elif args.env == 'flight_easy':
        args = get_flight_easy_args(args)
        targets_root = './flight_targets.txt'
        CIRCLE_DICT_2 = load_targets(targets_root)
        env = FlightSearchEnvEasy(args, circle_dict=CIRCLE_DICT_2)
    elif args.env == 'test':
        args = get_flight_args(args)
        env = TestSearchEnv()

    # 加载环境参数
    args = get_flight_easy_args(args)
    # 指定搜索目标的文件的路径
    targets_root = './flight_targets.txt'
    # 从包含搜索目标的文件加载目标信息
    CIRCLE_DICT_2 = load_targets(targets_root)
    # 创建飞行搜索环境实例
    env = FlightSearchEnvEasy(args, circle_dict=CIRCLE_DICT_2)
    # 获取环境信息
    env_info = env.get_env_info()
    # 设置与环境相关的参数
    args.n_actions = env_info['n_actions']
    args.state_shape = env_info['state_shape']
    args.obs_shape = env_info['obs_shape']
    args.episode_limit = env_info['episode_limit']
    # 运行实验
    for i in range(1):
        # 创建实验运行器
        runner = Runner(env, args)
        # 如果是展示模型
        if args.show:
            # 如果不需要收集数据，进行回放
            if not args.experiment:
                runner.replay(1521)
            else:
                #如果需要收集实验数据
                model_idx = 25
                replay_times = 100
                print('using model {}, replay time {}'.format(model_idx, replay_times))
                runner.collect_experiment_data(num=model_idx, replay_times=replay_times)
        else:
            #如果不是展示模式，运行实验
            runner.run(i)

        # 关闭环境
        env.close()

#     # 加载目标数量数据
#     targets_find_file = os.path.join(result_path, 'targets_find_{}.npy'.format(0))
#     if os.path.exists(targets_find_file):
#         targets_find = np.load(targets_find_file)
#         print("Targets found ({}):".format(targets_find.shape))
#         print(targets_find)
#
#     # 加载回报数据
#     episode_rewards_file = os.path.join(result_path, 'episode_rewards_{}.npy'.format(0))
#     if os.path.exists(episode_rewards_file):
#         episode_rewards = np.load(episode_rewards_file)
#         print("\nEpisode rewards ({}):".format(episode_rewards.shape))
#         print(episode_rewards)
