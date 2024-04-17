import numpy as np


class RolloutWorker:
    def __init__(self, env, agents, args):
        #从环境获得各参数
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = 0.05
        self.anneal_epsilon = 0.00064
        self.min_epsilon = 0.02
        print('Init RolloutWorker')

    '''这是一个生成一个episode的方法，负责智能体与环境的交互，收集经验。下面是对这个方法的概要：
    初始化：根据模式设置epsilon初始值，用于探索-利用决策。初始化用于存储observation、action、reward、state、可用动作、动作的独热编码、终止标志和填充标志的空列表。重置环境，初始化终止标志、胜利标志、步数、累积奖励和上一步动作。
    探索与决策：在每个时间步，智能体根据当前的观察和上一步动作选择下一步的动作。
    探索 - 利用策略根据当前epsilon值决定是以探索的方式选择动作，还是以利用的方式选择动作。选择的动作会被转化为动作的独热编码，并记录到相应的列表中。
    与环境交互：将选择的动作发送给环境，获得环境返回的奖励、终止信息以及其他相关信息。同时更新环境的内部状态，判断当前回合是否终止。如果终止，检查是否胜利。
    信息收集：将每个时间步的观察、状态、动作、动作的独热编码、可用动作、奖励、终止标志和填充标志记录到相应的列表中，用于后续的训练。
    终止条件：根据配置的终止条件或达到最大步数，循环结束。在循环结束后，如果是搜索环境，获取目标找到的数量。
    最后状态处理：获取最后一个时间步的观察和状态，并在之后处理这两个列表以得到下一个时间步的子序列。
    Padding：如果当前步数小于episode的时间上限，对episode进行padding，确保每个episode都具有相同的时间步数。
    字典生成：将所有收集到的信息整合到一个字典中，确保每个值都包含一个额外的episode维度。如果在评估模式下，关闭环境。
    返回结果：返回生成的episode，包括观察、状态、动作、奖励等信息，以及奖励的累积值和是否胜利的标志。如果是搜索环境，还返回目标找到的数量。'''

    #生成一个 episode 的数据。
    def generate_episode(self, episode_num=None, evaluate=False):
        #在评估模式下，如果是第一个 episode，则关闭环境。
        #为了准备保存评估过程的重播（replay）或清理之前的评估环境状态，以确保评估开始时环境是重新初始化的状态。
        if evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        #初始化用于存储 observation、action、reward、state、可用动作、动作的独热编码、终止标志和填充标志的空列表。
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        #重置环境，初始化终止标志、胜利标志、步数、累积奖励和上一步动作。
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        #对智能体的策略进行初始化，其中 1 表示 batch size 为 1。
        self.agents.policy.init_hidden(1)

        #根据模式设置 epsilon 初始值，如果是评估模式则设为 0，否则使用设定的 epsilon
        epsilon = 0 if evaluate else self.epsilon
        #随着每个episode的进行，逐渐减少探索的频率
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        #每个epoch开始时，模型会减少一次探索频率，而在该epoch内的其他episodes中保持ε不变。
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        #在 episode 未终止且步数未达到上限时进行循环。
        while not terminated and step < self.episode_limit:
            #获取环境的观察和状态。
            obs = self.env.get_obs()
            state = self.env.get_state()
            #初始化用于存储动作、可用动作和动作独热编码的空列表。
            actions, avail_actions, actions_onehot = [], [], []
            # if step % 100 == 0:
            #     print(evaluate, step, epsilon)
            #对每个智能体选择动作，根据 epsilon 使用智能体的策略进行选择。
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # print(obs.shape, last_action.shape)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                   epsilon, evaluate)

                # 将当前智能体选择的动作 action 添加到动作列表 actions 中。
                actions.append(action)
                #创建一个大小为 n_actions 的全零数组，用于表示动作的独热编码（one-hot encoding）
                action_onehot = np.zeros(self.args.n_actions)
                #将动作 action 对应的位置设为 1，得到独热编码
                action_onehot[action] = 1
                #将得到的独热编码 action_onehot 添加到独热编码列表 actions_onehot 中。
                actions_onehot.append(action_onehot)
                #将当前智能体可选的动作列表（在环境中可执行的动作）添加到 avail_actions 中。
                avail_actions.append(avail_action)
                #将当前智能体选择的动作的独热编码赋值给 last_action 中对应智能体的位置，以便在下一步中使用
                last_action[agent_id] = action_onehot
           #执行这个动作得到系统返回的奖励、是否终止、win_flag
            reward, terminated, info = self.env.step(actions)
            #如果当前回合终止且额外信息表明智能体取得了胜利，将 win_tag 设为 True
            win_tag = True if terminated and info else False
            #将观察、状态、动作、动作独热编码、可用动作、奖励、终止标志和填充标志添加到相应的列表中
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            #更新累积奖励和步数。
            #print("episode_reward:",episode_reward.shape)
            #print("reward:", reward.shape)
            for i in reward:
                episode_reward += i
            step += 1
            #如果 epsilon 的降低探索率度是按照步数进行的，则在每一步降低一次。
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        #如果是搜索环境，获取目标找到的数量。
        if self.args.search_env:
            targets_find = self.env.target_find

        #获取当前环境的观察信息，即智能体当前所处的状态。这个观察信息包括了所有智能体的局部观察
        obs = self.env.get_obs()
        #获取当前环境的全局状态信息
        state = self.env.get_state()
        #将当前观察信息添加到 episode 的观察列表中。
        o.append(obs)
        #将当前全局状态信息添加到 episode 的状态列表中
        s.append(state)
        #分别得到观察列表和状态列表的下一个时间步的子序列，用于构建下一个时间步的输入。
        o_next = o[1:]
        s_next = s[1:]
        #去除观察列表和状态列表的最后一个元素，这是因为上面已经获得了下一个时间步的子序列，所以当前时间步的信息不再需要。
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        #初始化一个列表，用于存储每个智能体在当前时间步的可执行动作
        avail_actions = []
        #循环遍历每个智能体，获取每个智能体在当前时间步的可执行动作。
        for agent_id in range(self.n_agents):
            #将每个智能体的可执行动作添加到 avail_actions 列表中。
            avail_action = self.env.get_avail_agent_actions(agent_id)
            #将每个智能体的可执行动作添加到 avail_actions 列表中。。
            avail_actions.append(avail_action)
        #将当前时间步的所有智能体的可执行动作列表添加到 episode 的可执行动作列表中。
        avail_u.append(avail_actions)
        #获取可执行动作列表的下一个时间步的子序列，为下一个时间步的输入准备数据。
        avail_u_next = avail_u[1: ]
        #去除avail_u 列表的最后一个元素。已经通过 avail_u_next 获取了下一个时间步的可执行动作列表，不需要当前时间步的可执行动作列表。
        avail_u = avail_u[:-1]

        if self.args.conv:
            obs_shape = self.obs_shape + self.args.map_size ** 2
        else:
            obs_shape = self.obs_shape

        # if step < self.episode_limit，padding
        #从当前步数step开始，一直循环到episode的时间上限，对episode进行padding的过程，确保每个episode都具有相同的时间步数
        for i in range(step, self.episode_limit):
            #向各个列表添加一个全零的数组
            o.append(np.zeros((self.n_agents, obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([[0. for _ in range(self.n_agents)]])
            o_next.append(np.zeros((self.n_agents, obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
        #创建一个字典episode，其中包含了当前episode的各个信息，如观察（o）、状态（s）、动作（u）、奖励（r）等。
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        #遍历episode字典的所有键。 add episode dim
        for key in episode.keys():
            #对于字典中的每个值，将外部包裹一层数组，表示增加一个episode的维度。
            episode[key] = np.array([episode[key]])
        #如果不是在评估模式下，将当前的epsilon值更新为训练中计算得到的值。
        if not evaluate:
            self.epsilon = epsilon
        #如果是在评估模式下，并且当前的episode是评估的最后一个episode，关闭环境。
        if evaluate and episode_num == self.args.evaluate_epoch - 1:
            self.env.close()
        #如果是search_env，则返回包含额外信息（win_tag和targets_find）的结果。
        if self.args.search_env:
            return episode, episode_reward, win_tag, targets_find
        else:
            return episode, episode_reward, False, False


    # show the model
    def generate_replay(self, force=0, render=True, collect=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # 重置环境，init=True 表示进行环境的初始化。
        self.env.reset(init=True)
        #初始化一些变量，包括回合是否终止、是否获胜、步数、累积奖励以及上一步的动作。
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        #对智能体的策略进行初始化
        self.agents.policy.init_hidden(1)
        #初始化探索率、评估标志以及一个列表 res 用于存储每一步智能体找到目标的比例。
        epsilon = 0
        evaluate = True
        res = []
        #直到回合终止或者达到了设定的回合长度上限。
        while not terminated and step < self.episode_limit:
            #获取环境的观察和状态信息。
            obs = self.env.get_obs()
            state = self.env.get_state()
            #初始化存储动作、可用动作和动作独热编码的列表。
            actions, avail_actions, actions_onehot = [], [], []
            #遍历每个智能体
            for agent_id in range(self.n_agents):
                #获取当前智能体可执行的动作。
                avail_action = self.env.get_avail_agent_actions(agent_id)
                #根据智能体的策略选择动作
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                   epsilon, evaluate)
                # generate onehot vector of th action
                #初始化一个大小为动作空间大小的全零数组用于动作的独热编码。
                action_onehot = np.zeros(self.args.n_actions)
                # 将选择的动作位置设为 1，得到动作的独热编码
                action_onehot[action] = 1
                #将动作、可用动作和动作独热编码添加到相应的列表中。
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            # act_list = [a.detach().cpu().numpy().reshape(-1)[0] for a in actions]
            # 执行动作，获取奖励、回合是否终止以及其他信息。
            reward, terminated, info = self.env.step(actions)
            #渲染环境
            if render:
                self.env.render()
            # 如果回合终止，则设置 win_tag 为 True。
            win_tag = True if terminated else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            print(reward)
            for i in reward:
                episode_reward += i#累积奖励。
            step += 1#步数+1
            # 获取目标找到的数量
            tgt_find = self.env.target_find
            #计算并存储目标找到的比例。
            res.append(tgt_find / self.args.target_num)

        # if not collect:
        #     # for i in range(len(res)):
        #         # print('{} steps, agents found {:.2f}% targets'.format(time_step_standard[i], res[i]*100))
        #     print('{} steps, agents found all targets'.format(step))
        #列表 res 用于存储每一步智能体找到目标的比例。确保 res 列表的长度等于self.episode_limit，即确保记录了每一步的目标找到比例，保持一致性
        for _ in range(self.episode_limit - len(res)):
            res.append(1.0)
        #如果是搜索环境，返回目标找到的数量、累积奖励、每步找到目标的比例和总步数。
        if self.args.search_env:
            targets_find = self.env.target_find
            return targets_find, episode_reward, res, step
        else:
            return 0, episode_reward, res, step
