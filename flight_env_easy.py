import numpy as np
import matplotlib.pyplot as plt


class Target:
    def __init__(self, pos, priority):
        self.pos = pos
        self.priority = priority
        self.find = False


class FlightSearchEnvEasy:

    def __init__(self, args, circle_dict):
        # env params
        self.args = args
        self.map_size = args.map_size# 地图的大小50
        self.target_num = args.target_num#目标数量15
        self.target_mode = args.target_mode#0
        self.agent_mode = args.agent_mode#0
        self.n_agents = args.n_agents#智能体数量3
        self.view_range = args.view_range#智能体的视野范围7
        self.circle_dict = circle_dict
        self.time_limit = args.time_limit#时间限制200
        self.turn_limit = args.turn_limit# 转向限制np.pi/4
        self.detect_prob = args.detect_prob#检测概率0.9
        self.wrong_alarm_prob = args.wrong_alarm_prob#误报概率0.1
        self.safe_dist = args.safe_dist# 安全距离1
        self.velocity = args.agent_velocity#速度1
        self.force_dist = args.force_dist#力的作用范围3
        self.n_actions = 3#动作数量3
        self.state_shape = self.n_agents * 4 + self.target_num * 3  # agent: x,y,cos(yaw),sin(yaw) target: x,y,find
        # self.obs_shape = self.map_size**2 + 5 # x,y,cos(yaw),sin(yaw),find_num + freq_map
        self.obs_shape = 4
        print('Init Env ' + args.env + ' {}a{}t(agent mode:{}, target mode:{})'.format(self.n_agents, self.target_num,
                                                                                       self.agent_mode,
                                                                                       self.target_mode))

        # env variables
        self.time_step = 0
        self.target_list = []
        self.target_pos = []
        self.target_find = 0
        # self.wrong_alarm = 0
        self.agent_pos = []
        self.agent_yaw = []
        self.total_reward = 0
        self.curr_reward = [0] * self.n_agents
        self.obs = []
        self.win_flag = False
        self.out_flag = []  # flag = 1 only when agent get out of the map是否出界
        self.target_map = dict()

        # reward
        self.FIND_ONE_TGT = 10#找到一个目标的奖励
        self.FIND_ALL_TGT = 100#找到所有目标的奖励
        self.OUT_PUNISH = -3#出界的惩罚 原来是-1
        self.MOVE_COST = -1#每步移动的代价

        # potential force factor潜在力（potential force）的因子
        self.POTENTIAL_FORCE_FACTOR = 0.8

        # init env
        self.reset(init=True)
        # self.get_state()

    #返回包含有关环境的信息的字典
    def get_env_info(self):
        env_info = {}
        env_info['n_actions'] = self.n_actions
        env_info['state_shape'] = self.state_shape
        env_info['obs_shape'] = self.obs_shape
        env_info['episode_limit'] = self.time_limit
        return env_info

    #重置环境到初始状态。根据 target_mode 和 agent_mode 初始化目标和智能体的位置。
    def reset(self, init=False):
        self.target_map = dict()#目标地图
        self.time_step = 0#时间步数
        self.target_find = 0#找到的目标数量
        # self.wrong_alarm = 0
        self.total_reward = 0#总奖励
        self.curr_reward = 0#当前奖励
        self.obs.clear()#观测列表
        self.target_list.clear()#目标列表
        self.target_pos.clear()#目标位置列表
        self.agent_pos.clear()#智能体位置列表
        self.agent_yaw.clear()#智能体方向列表
        self.out_flag.clear()#出界标志列表
        self.win_flag = False

        #从预设文件circle_dict中加载。
        if self.target_mode == 0:
            #为目标的各个属性赋值
            for i in range(self.target_num):
                a = self.map_size / 10
                x = a * self.circle_dict['x'][i]
                y = a * self.circle_dict['y'][i]
                deter = self.circle_dict['deter'][i]
                priority = self.circle_dict['priority'][i]
                dx = a * self.circle_dict['dx'][i]
                dy = a * self.circle_dict['dy'][i]
                if deter == 't':#静态目标
                    target_tmp = Target([x, y], priority)
                elif deter == 'f':#移动目标
                    #引入随机扰动 delta_x 和 delta_y，更新目标的位置
                    delta_x = dx * 2 * (np.random.randn() - 0.5)
                    delta_y = dy * 2 * (np.random.randn() - 0.5)
                    x += delta_x
                    y += delta_y
                    target_tmp = Target([x, y], priority)
                #将目标实例 target_tmp 添加到目标列表 target_list 中，并将目标的位置 [x, y] 添加到目标位置列表 target_pos 中。
                self.target_list.append(target_tmp)
                self.target_pos.append([x, y])

                # target map 确保坐标不超过地图的边界
                x_idx, y_idx = min(int(x), self.map_size - 1), min(int(y), self.map_size - 1)
                #如果不在target_map，表示这是新目标在这个位置，将一个包含当前目标索引 i 的列表作为值
                if (x_idx, y_idx) not in self.target_map:
                    self.target_map[(x_idx, y_idx)] = [i]
                else:
                    self.target_map[(x_idx, y_idx)].append(i)

        #完全随机生成目标的位置
        elif self.target_mode == 1:  # totally random
            for i in range(self.target_num):
                x, y = [self.map_size * np.random.rand() for _ in range(2)]
                target_tmp = Target([x, y], 1)
                self.target_list.append(target_tmp)
                self.target_pos.append([x, y])

                # target map
                x_idx, y_idx = min(int(x), self.map_size - 1), min(int(y), self.map_size - 1)
                if (x_idx, y_idx) not in self.target_map:
                    self.target_map[(x_idx, y_idx)] = [i]
                else:
                    self.target_map[(x_idx, y_idx)].append(i)
        else:
            raise Exception('No such target mode')

        # reset agents
        if self.agent_mode == 0 :  # start from the bottom line(left corner, medium, right corner)
            #如果agent数量不为1，使用列表解析生成智能体的横坐标 x，均匀分布在地图底线上
            if self.n_agents != 1:
                x = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
            #智能体数量为1，将智能体的横坐标 x 设置为地图的中心。
            else:
                x = [self.map_size / 2]
            y = 0
            #初始化智能体的位置、方向（假设为向上，即0度），以及出界标志（初始化为0）。
            for i in range(self.n_agents):
                self.agent_pos.append([x[i], y])
                self.agent_yaw.append(np.pi / 2)
                self.out_flag.append(0)
        elif self.agent_mode == 1:  # start from the medium of the map
            if self.n_agents != 1:
                x = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
            else:
                x = [self.map_size / 2]
            y = self.map_size / 2
            for i in range(self.n_agents):
                self.agent_pos.append([x[i], y])
                self.agent_yaw.append(np.pi / 2)
                self.out_flag.append(0)
        elif self.agent_mode == 2:  # start from the left line
            if self.n_agents != 1:
                y = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
            else:
                y = [self.map_size / 2]
            x = 0
            for i in range(self.n_agents):
                self.agent_pos.append([x, y[i]])
                self.agent_yaw.append(0)
                self.out_flag.append(0)
        elif self.agent_mode == 3:  # start from the right line
            if self.n_agents != 1:
                y = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
            else:
                y = [self.map_size / 2]
            x = self.map_size
            for i in range(self.n_agents):
                self.agent_pos.append([x, y[i]])
                self.agent_yaw.append(np.pi)
                self.out_flag.append(0)
        else:
            raise Exception('No such agent mode')

        self._update_obs()

    #获取给定智能体的可用动作。它接受一个智能体的ID作为输入，并返回一个数组，表示该智能体可以执行的动作。
    def get_avail_agent_actions(self, agent_id):
        #传入的智能体id 大于等于环境中智能体的数量n_agents，则抛出异常
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        #创建一个长度为 n_actions（3）的数组 avail_actions，并将其所有元素初始化为1
        avail_actions = np.ones(self.n_actions)
        return avail_actions

    #获取当前环境的全局状态，它是一个包含所有智能体和目标信息的合并向量。
    def get_state(self):
        # agent_state (x,y,cos,sin) shape:(n,4)
        agent_state = np.array(self.obs)
        #将智能体状态中每个智能体的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
        agent_state[:, :2] = (agent_state[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)

        target_state = []
        for i in range(self.target_num):
            s = []
            #将目标的位置信息（x和y坐标）添加到列表中。
            s.extend(self.target_pos[i])
            #目标被发现则加入1，否则为0
            if self.target_list[i].find:
                s.append(1.0)
            else:
                s.append(0.0)
            target_state.append(s)

        #将目标状态中每个目标的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
        target_state = np.array(target_state)
        target_state[:, :2] = (target_state[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)

        #将智能体状态矩阵和目标状态矩阵水平堆叠成一个状态向量 state，其中智能体状态矩阵被展平为一维数组，目标状态矩阵被展平为一维数组，然后两者被水平堆叠
        state = np.hstack([agent_state.reshape(4 * self.n_agents), target_state.reshape(3 * self.target_num)])
        #print('state:',state.shape, '\n', state)
        return state  # shape (4*n+3*m, )

    #获取当前环境中每个智能体的个体观测。
    def get_obs(self):
        obs = np.array(self.obs)
        #对观察数据中每个智能体的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
        obs[:, :2] = (obs[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)
        return obs

    #更新环境的观察值。计算每个智能体的当前位置、方向，并检查是否发现目标或智能体是否出界。
    def _update_obs(self):
        #清空观察数据列表 obs，并将当前奖励 curr_reward 初始化为0。
        self.obs.clear()

        self.curr_reward = [0] * self.n_agents

        #获取智能体的位置坐标x和y，以及偏航角yaw。
        for i in range(self.n_agents):

            self.curr_reward[i] += self.MOVE_COST
            x, y = self.agent_pos[i]
            yaw = self.agent_yaw[i]

            # find target reward
            #enumerate(self.target_list)内置函数enumerate用于同时迭代目标列表 self.target_list 中的元素及其索引。
            #返回第一个元素是目标在列表中的索引，第二个元素是目标对象本身。
            for j, target in enumerate(self.target_list):
                t_x, t_y = target.pos
                #计算智能体与目标之间的距离，如果距离小于等于视野范围 view_range 的平方，则表示目标在视野范围内。
                if (t_x - x) ** 2 + (t_y - y) ** 2 <= self.view_range ** 2:
                    prob = np.random.rand()  # misdetect
                    #如果目标未被发现且随机概率小于等于检测概率 detect_prob，则将目标标记为已发现
                    if not target.find and prob <= self.detect_prob:
                        target.find = True
                        #增加奖励
                        self.curr_reward[i] += self.FIND_ONE_TGT
                        self.target_find += 1
                        #检查是否所有目标都已被发现，
                        # 当所有目标被发现时，每个智能体都获得额外奖励
                        for k in range(self.n_agents):
                            self.curr_reward[k] += self.FIND_ALL_TGT / self.n_agents  # 均分 FIND_ALL_TGT 奖励IND_ALL_TGT
                            self.win_flag = True

            # 如果智能体出界 更新奖励
            if self.out_flag[i]:
                self.curr_reward[i] += self.OUT_PUNISH
            #创建包含智能体位置和方向信息的观察数据 obs，并将其添加到观察数据列表 self.obs 中。
            obs = np.array([x, y, np.cos(yaw), np.sin(yaw)])
            self.obs.append(obs)

    #根据提供的动作列表 act_list，更新每个智能体的位置和方向。考虑潜在力（以避免碰撞）和是否出界。
    def _agent_step(self, act_list):
        #如果传入的动作列表的长度与智能体的数量不匹配，抛出异常。
        if len(act_list) != self.n_agents:
            raise Exception('Act num mismatch agent')
        #定义了一个包含三种偏航角变化的列表 dyaw，用于根据动作调整智能体的偏航角。
        dyaw = [0, np.pi / 18, -np.pi / 18]
        for i, (x, y) in enumerate(self.agent_pos):
            #根据智能体的动作，调整智能体的偏航角。
            yaw = self.agent_yaw[i]
            yaw += dyaw[act_list[i]]  # change yaw of agent
            #如果调整后的偏航角超过了 0 到 2π 的范围，进行调整。
            if yaw > 2 * np.pi:  # yaw : 0~2pi
                yaw -= 2 * np.pi
            elif yaw < 0:
                yaw += 2 * np.pi
           #根据智能体的速度和偏航角更新智能体的位置。
            x += self.velocity * np.cos(yaw)
            y += self.velocity * np.sin(yaw)

            # add potential-energy function to avoid collision
            f_x, f_y = self._potential_energy_force(i)
            x += f_x
            y += f_y
            # x = min(max(x, 0), self.map_size)
            # y = min(max(y, 0), self.map_size)

            # check whether go out of the map检查是否超出地图范围
            if x < 0 or x > self.map_size or y < 0 or y > self.map_size:
                #如果智能体超出地图范围，则将其位置调整到地图边界
                x = min(max(x, 0), self.map_size)
                y = min(max(y, 0), self.map_size)
                #调整偏航角
                if yaw <= np.pi:
                    yaw = np.pi - yaw
                else:
                    yaw = 3 * np.pi - yaw
                #出界标志 out_flag 设置为1
                self.out_flag[i] = 1
            else:
                self.out_flag[i] = 0

            self.agent_pos[i] = [x, y]
            self.agent_yaw[i] = yaw
            # print(i, x, y, yaw)

   #计算和应用潜在能量力。当智能体接近彼此时，它们会互相施加力来避免碰撞。
    def _potential_energy_force(self, index):  # potential-energy force
        #获取当前智能体的位置坐标，并初始化力的初始值为0。
        x, y = self.agent_pos[index]
        f_x, f_y = 0, 0
        #对于每个智能体
        for i, (x_a, y_a) in enumerate(self.agent_pos):
            #如果智能体不是当前智能体，并且与当前智能体的距离小于规定的力的作用范围
            if i != index and (x_a - x) ** 2 + (y_a - y) ** 2 < self.force_dist ** 2:
                #如果其他智能体不与当前智能体位置完全重合，则计算受到的潜在能量函数的力。这个力与其他智能体之间的距离成反比，方向指向其他智能体。
                if x_a != x or y_a != y:
                    f_x += self.safe_dist * self.POTENTIAL_FORCE_FACTOR * self.velocity * (x - x_a) / (
                                (x - x_a) ** 2 + (y - y_a) ** 2)
                    f_y += self.safe_dist * self.POTENTIAL_FORCE_FACTOR * self.velocity * (y - y_a) / (
                                (x - x_a) ** 2 + (y - y_a) ** 2)
        return f_x, f_y

    #执行一个时间步骤。根据智能体的动作更新环境状态，返回当前奖励、是否终止和是否胜利
    def step(self, act_list):
        terminated = False
        #执行动作，更新智能体的位置。
        self._agent_step(act_list)
        #更新环境的观测值，奖励
        self._update_obs()
        self.total_reward += sum(self.curr_reward)
        self.time_step += 1
        #如果找到的目标数量达到了设定的目标数量target_num，或者时间步超过了设定的时间限制time_limit，则环境终止。
        if self.target_find >= self.target_num or self.time_step >= self.time_limit:
            terminated = True
        return self.curr_reward, terminated, self.win_flag


    #渲染当前环境状态。显示目标和智能体的位置。
    def render(self):
        #清除当前图形。
        plt.cla()
        #定义颜色数组，分别代表未发现目标、已发现目标和智能体。
        COLORS = ['black', 'green', 'orange']
        #对于每个目标，如果目标已经被发现，则用橙色标记，否则用黑色标记。
        for target in self.target_list:
            if target.find:
                plt.scatter(target.pos[0], target.pos[1], c=COLORS[2], s=7)
            else:
                plt.scatter(target.pos[0], target.pos[1], c=COLORS[0], s=7)
        #对于每个智能体，用红色三角形标记。
        for agent in self.agent_pos:
            [x, y] = agent
            plt.scatter(x, y, c='red', marker='^')
        title = 'target_find:{}/{}'.format(self.target_find, self.target_num)
        plt.title(title)
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.draw()
        #如果已发现的目标数量等于总目标数量，则暂停图形显示3秒
        if self.target_find == self.target_num:
            plt.pause(3)
        #否则暂停0.1秒。
        else:
            plt.pause(0.1)

    def close(self):
        pass

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class Target:
#     def __init__(self, pos, priority):
#         self.pos = pos
#         self.priority = priority
#         self.find = False
#
#
# class FlightSearchEnvEasy:
#
#     def __init__(self, args, circle_dict):
#         # env params
#         self.args = args
#         self.map_size = args.map_size# 地图的大小50
#         self.target_num = args.target_num#目标数量15
#         self.target_mode = args.target_mode#0
#         self.agent_mode = args.agent_mode#0
#         self.n_agents = args.n_agents#智能体数量3
#         self.view_range = args.view_range#智能体的视野范围7
#         self.circle_dict = circle_dict
#         self.time_limit = args.time_limit#时间限制200
#         self.turn_limit = args.turn_limit# 转向限制np.pi/4
#         self.detect_prob = args.detect_prob#检测概率0.9
#         self.wrong_alarm_prob = args.wrong_alarm_prob#误报概率0.1
#         self.safe_dist = args.safe_dist# 安全距离1
#         self.velocity = args.agent_velocity#速度1
#         self.force_dist = args.force_dist#力的作用范围3
#         self.n_actions = 3#动作数量3
#         self.state_shape = self.n_agents * 4 + self.target_num * 3  # agent: x,y,cos(yaw),sin(yaw) target: x,y,find
#         # self.obs_shape = self.map_size**2 + 5 # x,y,cos(yaw),sin(yaw),find_num + freq_map
#         self.obs_shape = 4
#         print('Init Env ' + args.env + ' {}a{}t(agent mode:{}, target mode:{})'.format(self.n_agents, self.target_num,
#                                                                                        self.agent_mode,
#                                                                                        self.target_mode))
#
#         # env variables
#         self.time_step = 0
#         self.target_list = []
#         self.target_pos = []
#         self.target_find = 0
#         # self.wrong_alarm = 0
#         self.agent_pos = []
#         self.agent_yaw = []
#         self.total_reward = 0
#         self.curr_reward = 0
#         self.obs = []
#         self.win_flag = False
#         self.out_flag = []  # flag = 1 only when agent get out of the map是否出界
#         self.target_map = dict()
#
#         # reward
#         self.FIND_ONE_TGT = 10#找到一个目标的奖励
#         self.FIND_ALL_TGT = 100#找到所有目标的奖励
#         self.OUT_PUNISH = -1#出界的惩罚
#         self.MOVE_COST = -1#每步移动的代价
#
#         # potential force factor潜在力（potential force）的因子
#         self.POTENTIAL_FORCE_FACTOR = 0.8
#
#         # init env
#         self.reset(init=True)
#         # self.get_state()
#
#     #返回包含有关环境的信息的字典
#     def get_env_info(self):
#         env_info = {}
#         env_info['n_actions'] = self.n_actions
#         env_info['state_shape'] = self.state_shape
#         env_info['obs_shape'] = self.obs_shape
#         env_info['episode_limit'] = self.time_limit
#         return env_info
#
#     #重置环境到初始状态。根据 target_mode 和 agent_mode 初始化目标和智能体的位置。
#     def reset(self, init=False):
#         self.target_map = dict()#目标地图
#         self.time_step = 0#时间步数
#         self.target_find = 0#找到的目标数量
#         # self.wrong_alarm = 0
#         self.total_reward = 0#总奖励
#         self.curr_reward = 0#当前奖励
#         self.obs.clear()#观测列表
#         self.target_list.clear()#目标列表
#         self.target_pos.clear()#目标位置列表
#         self.agent_pos.clear()#智能体位置列表
#         self.agent_yaw.clear()#智能体方向列表
#         self.out_flag.clear()#出界标志列表
#         self.win_flag = False
#
#         #从预设文件circle_dict中加载。
#         if self.target_mode == 0:
#             #为目标的各个属性赋值
#             for i in range(self.target_num):
#                 a = self.map_size / 10
#                 x = a * self.circle_dict['x'][i]
#                 y = a * self.circle_dict['y'][i]
#                 deter = self.circle_dict['deter'][i]
#                 priority = self.circle_dict['priority'][i]
#                 dx = a * self.circle_dict['dx'][i]
#                 dy = a * self.circle_dict['dy'][i]
#                 if deter == 't':#静态目标
#                     target_tmp = Target([x, y], priority)
#                 elif deter == 'f':#移动目标
#                     #引入随机扰动 delta_x 和 delta_y，更新目标的位置
#                     delta_x = dx * 2 * (np.random.randn() - 0.5)
#                     delta_y = dy * 2 * (np.random.randn() - 0.5)
#                     x += delta_x
#                     y += delta_y
#                     target_tmp = Target([x, y], priority)
#                 #将目标实例 target_tmp 添加到目标列表 target_list 中，并将目标的位置 [x, y] 添加到目标位置列表 target_pos 中。
#                 self.target_list.append(target_tmp)
#                 self.target_pos.append([x, y])
#
#                 # target map 确保坐标不超过地图的边界
#                 x_idx, y_idx = min(int(x), self.map_size - 1), min(int(y), self.map_size - 1)
#                 #如果不在target_map，表示这是新目标在这个位置，将一个包含当前目标索引 i 的列表作为值
#                 if (x_idx, y_idx) not in self.target_map:
#                     self.target_map[(x_idx, y_idx)] = [i]
#                 else:
#                     self.target_map[(x_idx, y_idx)].append(i)
#
#         #完全随机生成目标的位置
#         elif self.target_mode == 1:  # totally random
#             for i in range(self.target_num):
#                 x, y = [self.map_size * np.random.rand() for _ in range(2)]
#                 target_tmp = Target([x, y], 1)
#                 self.target_list.append(target_tmp)
#                 self.target_pos.append([x, y])
#
#                 # target map
#                 x_idx, y_idx = min(int(x), self.map_size - 1), min(int(y), self.map_size - 1)
#                 if (x_idx, y_idx) not in self.target_map:
#                     self.target_map[(x_idx, y_idx)] = [i]
#                 else:
#                     self.target_map[(x_idx, y_idx)].append(i)
#         else:
#             raise Exception('No such target mode')
#
#         # reset agents
#         if self.agent_mode == 0 :  # start from the bottom line(left corner, medium, right corner)
#             #如果agent数量不为1，使用列表解析生成智能体的横坐标 x，均匀分布在地图底线上
#             if self.n_agents != 1:
#                 x = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
#             #智能体数量为1，将智能体的横坐标 x 设置为地图的中心。
#             else:
#                 x = [self.map_size / 2]
#             y = 0
#             #初始化智能体的位置、方向（假设为向上，即0度），以及出界标志（初始化为0）。
#             for i in range(self.n_agents):
#                 self.agent_pos.append([x[i], y])
#                 self.agent_yaw.append(np.pi / 2)
#                 self.out_flag.append(0)
#         elif self.agent_mode == 1:  # start from the medium of the map
#             if self.n_agents != 1:
#                 x = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
#             else:
#                 x = [self.map_size / 2]
#             y = self.map_size / 2
#             for i in range(self.n_agents):
#                 self.agent_pos.append([x[i], y])
#                 self.agent_yaw.append(np.pi / 2)
#                 self.out_flag.append(0)
#         elif self.agent_mode == 2:  # start from the left line
#             if self.n_agents != 1:
#                 y = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
#             else:
#                 y = [self.map_size / 2]
#             x = 0
#             for i in range(self.n_agents):
#                 self.agent_pos.append([x, y[i]])
#                 self.agent_yaw.append(0)
#                 self.out_flag.append(0)
#         elif self.agent_mode == 3:  # start from the right line
#             if self.n_agents != 1:
#                 y = [i * self.map_size / (self.n_agents - 1) for i in range(self.n_agents)]
#             else:
#                 y = [self.map_size / 2]
#             x = self.map_size
#             for i in range(self.n_agents):
#                 self.agent_pos.append([x, y[i]])
#                 self.agent_yaw.append(np.pi)
#                 self.out_flag.append(0)
#         else:
#             raise Exception('No such agent mode')
#
#         self._update_obs()
#
#     #获取给定智能体的可用动作。它接受一个智能体的ID作为输入，并返回一个数组，表示该智能体可以执行的动作。
#     def get_avail_agent_actions(self, agent_id):
#         #传入的智能体id 大于等于环境中智能体的数量n_agents，则抛出异常
#         if agent_id >= self.n_agents:
#             raise Exception('Agent id out of range')
#         #创建一个长度为 n_actions（3）的数组 avail_actions，并将其所有元素初始化为1
#         avail_actions = np.ones(self.n_actions)
#         return avail_actions
#
#     #获取当前环境的全局状态，它是一个包含所有智能体和目标信息的合并向量。
#     def get_state(self):
#         # agent_state (x,y,cos,sin) shape:(n,4)
#         agent_state = np.array(self.obs)
#         #将智能体状态中每个智能体的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
#         agent_state[:, :2] = (agent_state[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)
#
#         target_state = []
#         for i in range(self.target_num):
#             s = []
#             #将目标的位置信息（x和y坐标）添加到列表中。
#             s.extend(self.target_pos[i])
#             #目标被发现则加入1，否则为0
#             if self.target_list[i].find:
#                 s.append(1.0)
#             else:
#                 s.append(0.0)
#             target_state.append(s)
#
#         #将目标状态中每个目标的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
#         target_state = np.array(target_state)
#         target_state[:, :2] = (target_state[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)
#
#         #将智能体状态矩阵和目标状态矩阵水平堆叠成一个状态向量 state，其中智能体状态矩阵被展平为一维数组，目标状态矩阵被展平为一维数组，然后两者被水平堆叠
#         state = np.hstack([agent_state.reshape(4 * self.n_agents), target_state.reshape(3 * self.target_num)])
#         #print('state:',state.shape, '\n', state)
#         return state  # shape (4*n+3*m, )
#
#     #获取当前环境中每个智能体的个体观测。
#     def get_obs(self):
#         obs = np.array(self.obs)
#         #对观察数据中每个智能体的前两个元素（x和y坐标）进行归一化，将其范围映射到 -1 到 1 之间。
#         obs[:, :2] = (obs[:, :2] - 0.5 * self.map_size) / (self.map_size / 2)
#         return obs
#
#     #更新环境的观察值。计算每个智能体的当前位置、方向，并检查是否发现目标或智能体是否出界。
#     def _update_obs(self):
#         #清空观察数据列表 obs，并将当前奖励 curr_reward 初始化为0。
#         self.obs.clear()
#         self.curr_reward = 0
#
#         #将移动成本增加到当前奖励
#         self.curr_reward += self.MOVE_COST
#         #获取智能体的位置坐标x和y，以及偏航角yaw。
#         for i in range(self.n_agents):
#             x, y = self.agent_pos[i]
#             yaw = self.agent_yaw[i]
#
#             # find target reward
#             #enumerate(self.target_list)内置函数enumerate用于同时迭代目标列表 self.target_list 中的元素及其索引。
#             #返回第一个元素是目标在列表中的索引，第二个元素是目标对象本身。
#             for j, target in enumerate(self.target_list):
#                 t_x, t_y = target.pos
#                 #计算智能体与目标之间的距离，如果距离小于等于视野范围 view_range 的平方，则表示目标在视野范围内。
#                 if (t_x - x) ** 2 + (t_y - y) ** 2 <= self.view_range ** 2:
#                     prob = np.random.rand()  # misdetect
#                     #如果目标未被发现且随机概率小于等于检测概率 detect_prob，则将目标标记为已发现
#                     if not target.find and prob <= self.detect_prob:
#                         target.find = True
#                         #增加奖励
#                         self.curr_reward += self.FIND_ONE_TGT
#                         self.target_find += 1
#                         #检查是否所有目标都已被发现，
#                         if self.target_find == self.target_num and not self.win_flag:
#                             self.curr_reward += self.FIND_ALL_TGT
#                             self.win_flag = True
#
#             # 如果智能体出界 更新奖励
#             if self.out_flag[i]:
#                 self.curr_reward += self.OUT_PUNISH
#             #创建包含智能体位置和方向信息的观察数据 obs，并将其添加到观察数据列表 self.obs 中。
#             obs = np.array([x, y, np.cos(yaw), np.sin(yaw)])
#             self.obs.append(obs)
#
#     #根据提供的动作列表 act_list，更新每个智能体的位置和方向。考虑潜在力（以避免碰撞）和是否出界。
#     def _agent_step(self, act_list):
#         #如果传入的动作列表的长度与智能体的数量不匹配，抛出异常。
#         if len(act_list) != self.n_agents:
#             raise Exception('Act num mismatch agent')
#         #定义了一个包含三种偏航角变化的列表 dyaw，用于根据动作调整智能体的偏航角。
#         dyaw = [0, np.pi / 18, -np.pi / 18]
#         for i, (x, y) in enumerate(self.agent_pos):
#             #根据智能体的动作，调整智能体的偏航角。
#             yaw = self.agent_yaw[i]
#             yaw += dyaw[act_list[i]]  # change yaw of agent
#             #如果调整后的偏航角超过了 0 到 2π 的范围，进行调整。
#             if yaw > 2 * np.pi:  # yaw : 0~2pi
#                 yaw -= 2 * np.pi
#             elif yaw < 0:
#                 yaw += 2 * np.pi
#            #根据智能体的速度和偏航角更新智能体的位置。
#             x += self.velocity * np.cos(yaw)
#             y += self.velocity * np.sin(yaw)
#
#             # add potential-energy function to avoid collision
#             f_x, f_y = self._potential_energy_force(i)
#             x += f_x
#             y += f_y
#             # x = min(max(x, 0), self.map_size)
#             # y = min(max(y, 0), self.map_size)
#
#             # check whether go out of the map检查是否超出地图范围
#             if x < 0 or x > self.map_size or y < 0 or y > self.map_size:
#                 #如果智能体超出地图范围，则将其位置调整到地图边界
#                 x = min(max(x, 0), self.map_size)
#                 y = min(max(y, 0), self.map_size)
#                 #调整偏航角
#                 if yaw <= np.pi:
#                     yaw = np.pi - yaw
#                 else:
#                     yaw = 3 * np.pi - yaw
#                 #出界标志 out_flag 设置为1
#                 self.out_flag[i] = 1
#             else:
#                 self.out_flag[i] = 0
#
#             self.agent_pos[i] = [x, y]
#             self.agent_yaw[i] = yaw
#             # print(i, x, y, yaw)
#
#    #计算和应用潜在能量力。当智能体接近彼此时，它们会互相施加力来避免碰撞。
#     def _potential_energy_force(self, index):  # potential-energy force
#         #获取当前智能体的位置坐标，并初始化力的初始值为0。
#         x, y = self.agent_pos[index]
#         f_x, f_y = 0, 0
#         #对于每个智能体
#         for i, (x_a, y_a) in enumerate(self.agent_pos):
#             #如果智能体不是当前智能体，并且与当前智能体的距离小于规定的力的作用范围
#             if i != index and (x_a - x) ** 2 + (y_a - y) ** 2 < self.force_dist ** 2:
#                 #如果其他智能体不与当前智能体位置完全重合，则计算受到的潜在能量函数的力。这个力与其他智能体之间的距离成反比，方向指向其他智能体。
#                 if x_a != x or y_a != y:
#                     f_x += self.safe_dist * self.POTENTIAL_FORCE_FACTOR * self.velocity * (x - x_a) / (
#                                 (x - x_a) ** 2 + (y - y_a) ** 2)
#                     f_y += self.safe_dist * self.POTENTIAL_FORCE_FACTOR * self.velocity * (y - y_a) / (
#                                 (x - x_a) ** 2 + (y - y_a) ** 2)
#         return f_x, f_y
#
#     #执行一个时间步骤。根据智能体的动作更新环境状态，返回当前奖励、是否终止和是否胜利
#     def step(self, act_list):
#         terminated = False
#         #执行动作，更新智能体的位置。
#         self._agent_step(act_list)
#         #更新环境的观测值，奖励
#         self._update_obs()
#         self.total_reward += self.curr_reward
#         self.time_step += 1
#         #如果找到的目标数量达到了设定的目标数量target_num，或者时间步超过了设定的时间限制time_limit，则环境终止。
#         if self.target_find >= self.target_num or self.time_step >= self.time_limit:
#             terminated = True
#
#         return self.curr_reward, terminated, self.win_flag
#
#     #渲染当前环境状态。显示目标和智能体的位置。
#     def render(self):
#         #清除当前图形。
#         plt.cla()
#         #定义颜色数组，分别代表未发现目标、已发现目标和智能体。
#         COLORS = ['black', 'green', 'orange']
#         #对于每个目标，如果目标已经被发现，则用橙色标记，否则用黑色标记。
#         for target in self.target_list:
#             if target.find:
#                 plt.scatter(target.pos[0], target.pos[1], c=COLORS[2], s=7)
#             else:
#                 plt.scatter(target.pos[0], target.pos[1], c=COLORS[0], s=7)
#         #对于每个智能体，用红色三角形标记。
#         for agent in self.agent_pos:
#             [x, y] = agent
#             plt.scatter(x, y, c='red', marker='^')
#         title = 'target_find:{}/{}'.format(self.target_find, self.target_num)
#         plt.title(title)
#         plt.xlim(0, self.map_size)
#         plt.ylim(0, self.map_size)
#         plt.draw()
#         #如果已发现的目标数量等于总目标数量，则暂停图形显示3秒
#         if self.target_find == self.target_num:
#             plt.pause(3)
#         #否则暂停0.1秒。
#         else:
#             plt.pause(0.1)
#
#     def close(self):
#         pass
