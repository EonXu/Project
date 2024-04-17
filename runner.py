import os
import numpy as np
import matplotlib.pyplot as plt
from agent import Agents
from rollout import RolloutWorker
from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(env, args) # 创建智能体 (Agents 类的实例)
        self.rolloutWorker = RolloutWorker(env, self.agents, args) # 创建 rollout worker (RolloutWorker 类的实例)
        #如果不是展示模型
        if not args.show:
            if args.off_policy: #如果这是off-policy
                self.buffer = ReplayBuffer(args, args.buffer_size)#初始化回放缓冲区

        self.args = args#保存接受的参数到类中
        self.win_rates = []  # 用于存储训练过程中的胜率
        self.targets_find = []  # 用于存储训练过程中找到目标的数量
        self.episode_rewards = []  # 用于存储训练过程中的回合奖励
        # 设置结果和模型的保存路径
        self.result_path = './result/' + args.env + '_Seed' + str(args.seed) + '_' + args.alg +\
                           '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
                                                                    args.target_mode)
        #如果路径不存在就创造此路径
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        #初始化用于存储模型的路径
        self.model_path = './model/' + args.env + '_Seed' + str(args.seed) + '_' + args.alg + \
                              '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
                                                args.target_mode)
        # 如果路径不存在就创造此路径
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def run(self, num):
        # 主训练循环
        train_steps = 0

        for epoch in range(100000):
            print('Run {}, train epoch {}'.format(num, epoch))
            # 判断当前训练周期是否是评估周期 200个epoch评估一次
            if epoch % 10 == 0:
            #if epoch % 200 == 0:
                #调用evaluate()方法进行模型的评估，返回三个值，分别是胜率 (win_rate)、平均回报 (episode_reward) 和目标找到的平均数量 (targets_find)。
                win_rate, episode_reward, targets_find = self.evaluate()
                if self.args.search_env:#如果需要搜索
                    print('Average targets found :{}/{}'.format(targets_find, self.args.target_num))
                    print('Average episode reward :{}'.format(episode_reward))
                else:#不需要搜索则只展示reward
                    print('Average episode reward :{}'.format(episode_reward))
                #将评估的值加入到列表中
                self.win_rates.append(win_rate)
                self.targets_find.append(targets_find)
                self.episode_rewards.append(episode_reward)

                self.plt(num)

            episodes = []
            # 收集1个episode
            for episode_idx in range(1):
                #调用 RolloutWorker 类的 generate_episode 方法，生成一个 episode。
                episode, _, _, _ = self.rolloutWorker.generate_episode(episode_idx)
                #将生成的 episode 添加到列表 episodes 中。
                episodes.append(episode)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            # 将episodes 列表中的第一个 episode 赋值给 episode_batch
            episode_batch = episodes[0]
            #从episodes列表中移除第一个 episode。这是因为已经将它赋值给了 episode_batch，不再需要在列表中保留。
            episodes.pop(0)
            #对于列表 episodes 中的每一个 episode
            for episode in episodes:
                for key in episode_batch.keys():
                    #将当前循环的 episode 中的键 key 对应的值，通过 np.concatenate
                    # 沿着指定轴（axis=0，表示在行的方向上拼接）拼接到 episode_batch 的对应键 key 的值中
                    # 这样，episode_batch 中的信息就逐渐包含了所有收集到的 episodes 的数据。
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            #如果不是离线学习模式：
            if not self.args.off_policy:
                #调用智能体的 train 方法，用整个 episode_batch 对智能体进行训练。
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                #train_steps 是一个计数器，记录训练的步数。
                train_steps += 1
            else:
                #是离线学习，将整个 episode_batch 存储到回放缓冲区中
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):#循环进行离线训练的步数。
                    #从回放缓冲区中采样一个小批次数据，
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        self.plt(num)

    def evaluate(self):
        #初始化三个变量，用于记录评估过程中的胜利次数 (win_number)、总回报 (episode_rewards) 和总目标找到数量
        win_number = 0
        episode_rewards = 0
        cumu_targets = 0
        #20 number of the epoch to evaluate the agents 进行20个独立的评估周期。
        for epoch in range(20):
            #调用 RolloutWorker 类的 generate_episode 方法，生成一个 episode，并传入参数 evaluate=True 表示当前是在评估模式下生成 episode。
            #返回的四个值分别是当前episode 的状态信息、episode 的总回报 (episode_reward)、是否获胜 (win_tag) 以及找到的目标数量 (targets_find)。
            _, episode_reward, win_tag, targets_find = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            cumu_targets += targets_find
            #如果当前episode获胜，胜利次数+1
            if win_tag:
                win_number += 1
        #返回3个平均值：胜利率、多轮episode的平均回报、多次查找的目标查找平均数量
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, cumu_targets / self.args.evaluate_epoch

    def plt(self, num):
        #创建一个新的图形。
        plt.figure()
        #设置坐标轴的范围，横坐标范围为 [0, self.args.n_epoch]，纵坐标范围为 [0, 100]。
        plt.axis([0, self.args.n_epoch, 0, 100])
        #清除当前图形的坐标轴。
        plt.cla()
        #如果是搜索环境
        if self.args.search_env:
            #创建一个子图，将找到目标的数量随着时间的变化绘制在子图中。
            plt.subplot(2, 1, 1)
            plt.plot(range(len(self.targets_find)), self.targets_find)
            plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
            plt.ylabel('targets_find')

            # 将找到目标的数量保存为NumPy数组文件（.npy格式），文件名为 'targets_find_{}.npy'，
            # 其中{}部分由参数num提供。保存的路径是在self.result_path下。
            np.save(os.path.join(self.result_path, 'targets_find_{}'.format(num)), self.targets_find)

            plt.subplot(2, 1, 2)
        #将所有 episode 的总回报随着时间的变化绘制在图中。
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        #将绘制的图表保存为文件，文件名为 'plt_{}.png'，其中 {} 部分由参数 num 提供。保存的路径是在 self.result_path 下。
        plt.savefig(os.path.join(self.result_path, 'plt_{}.png'.format(num)), format='png')
        #将所有episode的总回报保存为NumPy数组文件，文件名为 'episode_rewards_{}.npy'，其中 {} 部分由参数 num 提供。保存的路径是在 self.result_path 下
        np.save(os.path.join(self.result_path, 'episode_rewards_{}'.format(num)), self.episode_rewards)
        plt.close()

    #于加载预训练模型进行回放。
    def replay(self, num):  # the num of model loaded
        if self.args.alg == 'ddqn' or self.args.alg == 'd3qn':
            # 构建 DDQN模型的路径，加载 RNN 部分的预训练参数。
            ddqn_root = os.path.join(self.model_path, str(num) + '_ddqn_net_params.pkl')
            # 使用 self.agents.policy.load_model 方法加载 RNN 的预训练参数。
            self.agents.policy.load_model(ddqn_root)
        elif self.args.alg == 'reinforce':
            rnn_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
            self.agents.policy.load_model(rnn_root)

        else:
            raise Exception('Unknown Algorithm model to load')

        #调用 RolloutWorker 的 generate_replay 方法进行回放。
        #render=True 表示要在回放中进行可视化（渲染）。
        #获取回放过程中的 目标找到数量targets_find，回放的累积奖励episode_reward，回放结果res 和步数step。
        targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(render=True)
        print('targets_find: ', targets_find, ' reward: ', episode_reward)

    def collect_experiment_data(self, num, replay_times):
        #初始化四个空列表，分别用于存储目标找到数量、奖励、结果和步数的数据。
        tgt_find_list, reward_list, res_list, step_list = [], [], [], []
        for i in range(replay_times):
            # load model
            if self.args.alg == 'reinforce':
                actor_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
                self.agents.policy.load_model(actor_root)
            elif self.args.alg == 'ddqn' or 'd3qn':
                # 构建 DDQN 模型的路径
                ddqn_root = os.path.join(self.model_path, str(num) + '_ddqn_net_params.pkl')
                # 并加载预训练参数。
                self.agents.policy.load_model(ddqn_root)

            #调用 RolloutWorker 的 generate_replay 方法进行回放，设置 collect=True 表示要进行数据收集，render=False 表示不进行渲染。
            #获取回放过程中 目标找到数量targets_find，回放的累积奖励episode_reward，回放结果res 和步数step。
            targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(collect=True, render=False)
            #将每次实验的目标找到数量、奖励、结果和步数添加到对应的列表中。

            tgt_find_list.append(targets_find)
            reward_list.append(episode_reward)
            res_list.append(res)
            step_list.append(step)
            # print('experiment {} finished'.format(i))

        #计算目标找到数量、奖励和步数的平均值。
        average_tgt_find = np.mean(tgt_find_list)
        average_rew = np.mean(reward_list)
        average_step = np.mean(step_list)
        print(average_tgt_find, average_rew, average_step)
        #将结果的列表转换为 NumPy 数组，并计算结果的每一项的平均值。
        res_list = np.array(res_list)
        average_res = np.mean(res_list, axis=0)
        #打印结果的特定索引处的平均值，并乘以 100。
        idx_list = [10, 20, 40, 60, 80, 100, 150, 199]
        print(average_res[idx_list] * 100)
        #将平均结果保存为 NumPy 文件。
        np.save(os.path.join(self.result_path, 'average_res_{}'.format(num)), average_res * 100)
        print('process data saved!')
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from agent import Agents
# from rollout import RolloutWorker
# from replay_buffer import ReplayBuffer
#
#
# class Runner:
#     def __init__(self, env, args):
#         self.env = env
#         self.agents = Agents(env, args) # 创建智能体 (Agents 类的实例)
#         self.rolloutWorker = RolloutWorker(env, self.agents, args) # 创建 rollout worker (RolloutWorker 类的实例)
#         #如果不是展示模型
#         if not args.show:
#             if args.off_policy: #如果这是off-policy
#                 self.buffer = ReplayBuffer(args, args.buffer_size)#初始化回放缓冲区
#
#         self.args = args#保存接受的参数到类中
#         self.win_rates = []  # 用于存储训练过程中的胜率
#         self.targets_find = []  # 用于存储训练过程中找到目标的数量
#         self.episode_rewards = []  # 用于存储训练过程中的回合奖励
#         # 设置结果和模型的保存路径
#         self.result_path = './result/' + args.env + '_Seed' + str(args.seed) + '_' + args.alg +\
#                            '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
#                                                                     args.target_mode)
#         #如果路径不存在就创造此路径
#         if not os.path.exists(self.result_path):
#             os.makedirs(self.result_path)
#         #初始化用于存储模型的路径
#         self.model_path = './model/' + args.env + '_Seed' + str(args.seed) + '_' + args.alg + \
#                               '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
#                                                 args.target_mode)
#         # 如果路径不存在就创造此路径
#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#
#     def run(self, num):
#         # 主训练循环
#         train_steps = 0
#
#         for epoch in range(100000):
#             print('Run {}, train epoch {}'.format(num, epoch))
#             # 判断当前训练周期是否是评估周期 200个epoch评估一次
#             if epoch % 200 == 0:
#                 #调用evaluate()方法进行模型的评估，返回三个值，分别是胜率 (win_rate)、平均回报 (episode_reward) 和目标找到的平均数量 (targets_find)。
#                 win_rate, episode_reward, targets_find = self.evaluate()
#                 if self.args.search_env:#如果需要搜索
#                     print('Average targets found :{}/{}'.format(targets_find, self.args.target_num))
#                     print('Average episode reward :{}'.format(episode_reward))
#                 else:#不需要搜索则只展示reward
#                     print('Average episode reward :{}'.format(episode_reward))
#                 #将评估的值加入到列表中
#                 self.win_rates.append(win_rate)
#                 self.targets_find.append(targets_find)
#                 self.episode_rewards.append(episode_reward)
#
#                 self.plt(num)
#
#             episodes = []
#             # 收集1个episode
#             for episode_idx in range(1):
#                 #调用 RolloutWorker 类的 generate_episode 方法，生成一个 episode。
#                 episode, _, _, _ = self.rolloutWorker.generate_episode(episode_idx)
#                 #将生成的 episode 添加到列表 episodes 中。
#                 episodes.append(episode)
#                 # print(_)
#             # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
#             # 将episodes 列表中的第一个 episode 赋值给 episode_batch
#             episode_batch = episodes[0]
#             #从episodes列表中移除第一个 episode。这是因为已经将它赋值给了 episode_batch，不再需要在列表中保留。
#             episodes.pop(0)
#             #对于列表 episodes 中的每一个 episode
#             for episode in episodes:
#                 for key in episode_batch.keys():
#                     #将当前循环的 episode 中的键 key 对应的值，通过 np.concatenate
#                     # 沿着指定轴（axis=0，表示在行的方向上拼接）拼接到 episode_batch 的对应键 key 的值中
#                     # 这样，episode_batch 中的信息就逐渐包含了所有收集到的 episodes 的数据。
#                     episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
#             #如果不是离线学习模式：
#             if not self.args.off_policy:
#                 #调用智能体的 train 方法，用整个 episode_batch 对智能体进行训练。
#                 self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
#                 #train_steps 是一个计数器，记录训练的步数。
#                 train_steps += 1
#             else:
#                 #是离线学习，将整个 episode_batch 存储到回放缓冲区中
#                 self.buffer.store_episode(episode_batch)
#                 for train_step in range(self.args.train_steps):#循环进行离线训练的步数。
#                     #从回放缓冲区中采样一个小批次数据，
#                     mini_batch = self.buffer.sample(min(self.buffer.current_size, 32))
#                     self.agents.train(mini_batch, train_steps)
#                     train_steps += 1
#         self.plt(num)
#
#     def evaluate(self):
#         #初始化三个变量，用于记录评估过程中的胜利次数 (win_number)、总回报 (episode_rewards) 和总目标找到数量
#         win_number = 0
#         episode_rewards = 0
#         cumu_targets = 0
#         #20 number of the epoch to evaluate the agents 进行20个独立的评估周期。
#         for epoch in range(20):
#             #调用 RolloutWorker 类的 generate_episode 方法，生成一个 episode，并传入参数 evaluate=True 表示当前是在评估模式下生成 episode。
#             #返回的四个值分别是当前episode 的状态信息、episode 的总回报 (episode_reward)、是否获胜 (win_tag) 以及找到的目标数量 (targets_find)。
#             _, episode_reward, win_tag, targets_find = self.rolloutWorker.generate_episode(epoch, evaluate=True)
#             episode_rewards += episode_reward
#             cumu_targets += targets_find
#             #如果当前episode获胜，胜利次数+1
#             if win_tag:
#                 win_number += 1
#         #返回3个平均值：胜利率、多轮episode的平均回报、多次查找的目标查找平均数量
#         return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, cumu_targets / self.args.evaluate_epoch
#
#     def plt(self, num):
#         #创建一个新的图形。
#         plt.figure()
#         #设置坐标轴的范围，横坐标范围为 [0, self.args.n_epoch]，纵坐标范围为 [0, 100]。
#         plt.axis([0, self.args.n_epoch, 0, 100])
#         #清除当前图形的坐标轴。
#         plt.cla()
#         #如果是搜索环境
#         if self.args.search_env:
#             #创建一个子图，将找到目标的数量随着时间的变化绘制在子图中。
#             plt.subplot(2, 1, 1)
#             plt.plot(range(len(self.targets_find)), self.targets_find)
#             plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
#             plt.ylabel('targets_find')
#
#             # 将找到目标的数量保存为NumPy数组文件（.npy格式），文件名为 'targets_find_{}.npy'，
#             # 其中{}部分由参数num提供。保存的路径是在self.result_path下。
#             np.save(os.path.join(self.result_path, 'targets_find_{}'.format(num)), self.targets_find)
#
#             plt.subplot(2, 1, 2)
#         #将所有 episode 的总回报随着时间的变化绘制在图中。
#         plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
#         plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
#         plt.ylabel('episode_rewards')
#         #将绘制的图表保存为文件，文件名为 'plt_{}.png'，其中 {} 部分由参数 num 提供。保存的路径是在 self.result_path 下。
#         plt.savefig(os.path.join(self.result_path, 'plt_{}.png'.format(num)), format='png')
#         #将所有episode的总回报保存为NumPy数组文件，文件名为 'episode_rewards_{}.npy'，其中 {} 部分由参数 num 提供。保存的路径是在 self.result_path 下
#         np.save(os.path.join(self.result_path, 'episode_rewards_{}'.format(num)), self.episode_rewards)
#
#     #于加载预训练模型进行回放。
#     def replay(self, num):  # the num of model loaded
#         #对于reinforce 构建 REINFORCE 模型的路径，加载 RNN 部分的预训练参数。
#         rnn_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
#         #使用 self.agents.policy.load_model 方法加载 RNN 的预训练参数。
#         self.agents.policy.load_model(rnn_root)
#
#         #调用 RolloutWorker 的 generate_replay 方法进行回放。
#         #render=True 表示要在回放中进行可视化（渲染）。
#         #获取回放过程中的 目标找到数量targets_find，回放的累积奖励episode_reward，回放结果res 和步数step。
#         targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(render=True)
#         print('targets_find: ', targets_find, ' reward: ', episode_reward)
#
#     def collect_experiment_data(self, num, replay_times):
#         #初始化四个空列表，分别用于存储目标找到数量、奖励、结果和步数的数据。
#         tgt_find_list, reward_list, res_list, step_list = [], [], [], []
#         for i in range(replay_times):
#             # load model 'reinforce'
#             #构建 REINFORCE 模型的路径
#             actor_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
#             #并加载预训练参数。
#             self.agents.policy.load_model(actor_root)
#             #调用 RolloutWorker 的 generate_replay 方法进行回放，设置 collect=True 表示要进行数据收集，render=False 表示不进行渲染。
#             #获取回放过程中 目标找到数量targets_find，回放的累积奖励episode_reward，回放结果res 和步数step。
#             targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(collect=True, render=False)
#             #将每次实验的目标找到数量、奖励、结果和步数添加到对应的列表中。
#             tgt_find_list.append(targets_find)
#             reward_list.append(episode_reward)
#             res_list.append(res)
#             step_list.append(step)
#             # print('experiment {} finished'.format(i))
#         #计算目标找到数量、奖励和步数的平均值。
#         average_tgt_find = np.mean(tgt_find_list)
#         average_rew = np.mean(reward_list)
#         average_step = np.mean(step_list)
#         print(average_tgt_find, average_rew, average_step)
#         #将结果的列表转换为 NumPy 数组，并计算结果的每一项的平均值。
#         res_list = np.array(res_list)
#         average_res = np.mean(res_list, axis=0)
#         #打印结果的特定索引处的平均值，并乘以 100。
#         idx_list = [10, 20, 40, 60, 80, 100, 150, 199]
#         print(average_res[idx_list] * 100)
#         #将平均结果保存为 NumPy 文件。
#         np.save(os.path.join(self.result_path, 'average_res_{}'.format(num)), average_res * 100)
#         print('process data saved!')
