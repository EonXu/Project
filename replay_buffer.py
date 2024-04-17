import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, args, buffer_size):
        self.args = args
        self.n_actions = self.args.n_actions#智能体的动作数量
        self.n_agents = self.args.n_agents#智能体数量
        self.state_shape = self.args.state_shape#状态形状
        self.obs_shape = self.args.obs_shape#观测形状
        self.size = buffer_size#3000
        self.episode_limit = self.args.episode_limit#200
        # memory management
        self.current_idx = 0
        self.current_size = 0

        if self.args.conv:
            obs_shape = self.obs_shape + args.map_size ** 2
        else:
            obs_shape = self.obs_shape
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, obs_shape]),#当前观测
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),#动作
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),#状态
                        'r': np.empty([self.size, self.episode_limit,1,self.n_agents]),#奖励
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, obs_shape]),#下一个观测
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),#下一个状态
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),#可用动作
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),#下一个可用动作
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),#独热编码的动作
                        'padded': np.empty([self.size, self.episode_limit, 1]),#填充标志 标记数据是否被填充。在处理不同长度的序列时，填充是常见的做法。
                        'terminated': np.empty([self.size, self.episode_limit, 1])#终止标志
                        }
        print("size:",self.size)
        print("limit:",self.episode_limit)
        # thread lock 创建一个线程锁 lock，用于在多线程环境中保护共享资源。
        self.lock = threading.Lock()
        print('Init ReplayBuffer({})'.format(self.size))

        # store the episode

    def store_episode(self, episode_batch):
        #取传入回合数据中的批次大小，即回合的数量
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            #调用_get_storage_idx 方法获取存储的索引
            idxs = self._get_storage_idx(inc=batch_size)
            # store the information将传入回合数据中的各个信息存储到 ReplayBuffer 中
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    #确认缓冲区中是否有足够的数据进行采样。
    def can_sample(self, batch_size):
        if self.current_size >= batch_size:
            return True
        else:
            return False

    #随机选择哪些历史记录将被包含在采样的批次中
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)#生成batch_size个随机整数，这些整数的取值范围在current_size内。
        for key in self.buffers.keys():#对于每个键，将 self.buffers[key] 中对应位置索引为 idx 的数据提取出来，形成新的字典 temp_buffer。
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    #是从经验回放缓冲区中抽取最新的一批数据。
    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        idx = []
        if self.current_idx >= batch_size:
            idx = list(range(self.current_idx - batch_size, self.current_idx))
        else:
            left = batch_size - self.current_idx
            idx = list(range(self.current_size - left, self.current_size)) + list(range(self.current_idx))
        idx = np.array(idx)
        temp_buffer = {}
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        #如果没有提供增量 (inc)，默认设置为 1。
        inc = inc or 1
        #如果当前索引加上增量 (inc) 不超过缓冲区大小 (size)，直接返回从当前索引开始的一组索引，并更新当前索引。
        if self.current_idx + inc <= self.size:
            #np.arange()返回一个具有指定范围、步长和数据类型的一维数组。
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        #如果当前索引小于缓冲区大小，但是加上增量后会超过缓冲区大小
        elif self.current_idx < self.size:
            #将索引拆分成两部分：一部分从当前索引到缓冲区末尾，另一部分从缓冲区开头到溢出的部分
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            #然后拼接这两部分索引。
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        #如果当前索引已经到达缓冲区末尾，重新从索引 0 开始，返回从 0 开始的一组索引。
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        #更新当前经验回访数组的大小 (current_size)，确保不超过经验回访数组大小。
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
