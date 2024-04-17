import torch
import numpy as np
from policy.DDQN import DDQN
from policy.D3QN import D3QN
from policy.trandition import Random
from policy.reinforce import Reinforce
from torch.distributions import Categorical


class Agents():
    def __init__(self, env, args):
        self.map_size = args.map_size#50
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents#3
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.env = env
        self.args = args
        # 根据算法选择，初始化了对应的策略 (policy) 对象
        # if args.alg == 'qmix':
        #     self.policy = QMIX(args)
        # elif args.alg == 'dop':
        #     self.policy = DOP(args)
        if args.alg == 'reinforce':
            self.policy = Reinforce(args)
        elif args.alg == 'ddqn':
            self.policy = DDQN(args)
        elif args.alg == 'd3qn':
            self.policy = D3QN(args)
        elif args.alg == 'random':
            self.policy = Random(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        if self.args.alg == 'random':
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
        else:
            #复制观察
            inputs = obs.copy()
            avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
            #创建了一个长度为n_agents的独热编码向量agent_id
            agent_id = np.zeros(self.n_agents)
            #索引为agent_num的元素设置为1，表示当前代理。
            agent_id[agent_num] = 1.
            #如果args中设置了last_action标志，它使用np.hstack()将last_action数组水平堆叠到inputs数组中。
            if self.args.last_action:
                inputs = np.hstack((inputs, last_action))
            #如果args中设置了reuse_network标志，它将agent_id独热编码向量水平堆叠到inputs数组中。
            if self.args.reuse_network:
                inputs = np.hstack((inputs, agent_id))
            #从policy的eval_hidden属性中获取当前代理的隐藏状态，使用数组索引。
            hidden_state = self.policy.eval_hidden[:, agent_num, :]
            # transform the shape of inputs from (42,) to (1,42)
            #将inputs和avail_actions数组转换为PyTorch张量，使用unsqueeze(0)在每个张量上添加一个单例维度。
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
                hidden_state = hidden_state.cuda()

            #get q value 根据算法类型，从策略对象中获取Q值。对于'dop'算法，使用actor方法获取Q值，否则使用eval_rnn方法。
            if self.args.alg == 'dop':
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.actor(inputs, hidden_state)
            elif self.args.alg == 'ddqn' or self.args.alg == 'd3qn':
                # 将inputs和hidden_state张量传递给policy的eval_rnn方法，获取Q值(q_value)
                # 还更新policy的eval_hidden属性以反映当前代理的变化。
                q_values, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_ddqn(inputs, hidden_state)
                # print("Q Values:", q_values)
            else:
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

            # choose action from q value 根据算法类型选择动作。
            # 对于 'reinforce' 算法，使用 _choose_action_from_softmax 方法从 softmax 分布中选择动作。
            if self.args.alg == 'reinforce':
                action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
            # 对于其他算法，将 Q 值中不可选动作的概率设置为负无穷，然后根据贪婪策略或以概率 epsilon 随机选择动作。
            else:
                #3.8改正前
                # # choose action from q value 传递Q值、可用动作、epsilon和评估标志，然后返回选择的动作。
                # # 使用epsilon-greedy策略选择动作
                # action = self._choose_action_epsilon_greedy(q_values, avail_actions, epsilon, evaluate)

                #3.8改正
                q_values[avail_actions == 0.0] = - float("inf")
                if evaluate or np.random.rand() >= epsilon:
                    action = torch.argmax(q_values)
                else:
                    action = np.random.choice(avail_actions_ind)
                #print("Chosen Action:", action)
        return action

    # def _choose_action_epsilon_greedy(self, q_values, avail_actions, epsilon, evaluate=False):
    #     # epsilon-greedy 策略
    #     if np.random.random() < epsilon and not evaluate:
    #         avail_actions_idx = np.nonzero(avail_actions)[0]
    #         action = np.random.choice(avail_actions_idx)  # 随机选择可用动作
    #     else:
    #         q_values = q_values.cpu().detach().numpy().squeeze(0)
    #         q_values[avail_actions == 0] = -float('inf')  # 把不可用的动作置为负无穷
    #         action = np.argmax(q_values)  # 选择Q值最大的动作
    #     return action
    def _choose_action_epsilon_greedy(self, q_values, avail_actions, epsilon, evaluate=False):
        if epsilon > 0 and not evaluate and np.random.rand() < epsilon:
            # 随机选择一个可执行动作
            avail_actions_idx = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_idx)
        else:
            # 选择 Q 值最大的可执行动作
            q_values[avail_actions == 0] = -float('inf')  # 把不可用的动作置为负无穷
            action = torch.argmax(q_values).item()
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        #计算可执行动作的数量，
        #avail_actions.sum(dim=1, keepdim=True) 对每个样本计算可执行动作的总数，
        # .float() 将结果转换为浮点数，然后使用 .repeat(1, avail_actions.shape[-1]) 在最后一个维度上重复，以匹配 avail_actions 的形状。
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])
        # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布,dim=-1 表示在最后一个维度上进行 softmax 操作。
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon 引入探索噪音，通过在概率分布上加入噪音，即在原始概率上加权混合一个平均概率。
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        #如果 epsilon 为零且处于评估模式时，选择具有最大概率的动作
        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)#返回概率最大的动作的索引。
        else:
            #使用torch.distributions.Categorical 从概率分布中采样一个动作。
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        #从批次中获取包含终止信息的张量，表示每个 episode 是否结束。
        terminated = batch['terminated']
        #计算批次中的 episode 数量。
        episode_num = terminated.shape[0]
        #获取批次中 o（观察）的张量的第二个维度的大小，即每个 episode 的最大长度。
        max_episode_len = batch['o'].shape[1]
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):#遍历每个 episode 的每个 transition。
                if terminated[episode_idx, transition_idx, 0] == 1:
                    #如果当前 episode 的长度（transition_idx + 1）大于等于之前记录的最大长度，
                    #更新 max_episode_len 为当前 episode 的长度。
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    #一旦发现当前 episode 已经终止，跳出内层循环，继续下一个 episode 的检查。
                    break
        # print('1 ',max_episode_len)
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        #调用之前定义的 _get_max_episode_len 方法获取当前批次中所有 episode 的最大长度。
        max_episode_len = self._get_max_episode_len(batch)
        #遍历批次中的所有键（key 是字典 batch 中的键)
        for key in batch.keys():
            #截取每个键对应的值的第二个维度，将其限制为最大 episode 长度。
            batch[key] = batch[key][:, :max_episode_len]
        # 调用策略对象 (self.policy) 的 learn 方法，传递了截断后的批次数据、最大 episode 长度、训练步数以及贪婪策略参数。
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        #如果当前训练步数大于零且能够整除 self.args.save_cycle，则执行保存模型的操作。
        #通过调用 get_model_idx 获取模型的索引，然后调用 save_model 方法保存模型
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            idx = self.policy.get_model_idx()
            self.policy.save_model(idx)



# import torch
# import numpy as np
# from reinforce import Reinforce
# from torch.distributions import Categorical
#
#
# class Agents:
#     def __init__(self, env, args):
#         self.map_size = args.map_size#50
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents#3
#         self.state_shape = args.state_shape
#         self.obs_shape = args.obs_shape
#         self.env = env
#         self.policy = Reinforce(args)
#         self.args = args
#         print('Init Agents')
#
    # def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
    #     #复制观察
    #     inputs = obs.copy()
    #     #使用np.nonzero()找到avail_actions数组中非零元素的索引，将结果保存在avail_actions_ind中。
    #     avail_actions_ind = np.nonzero(avail_actions)[0]
    #     # index of actions which can be chosen
    #     # transform agent_num to onehot vector
    #     #创建了一个长度为n_agents的独热编码向量agent_id
    #     agent_id = np.zeros(self.n_agents)
    #     #索引为agent_num的元素设置为1，表示当前代理。
    #     agent_id[agent_num] = 1.
    #     #如果args中设置了last_action标志，它使用np.hstack()将last_action数组水平堆叠到inputs数组中。
    #     if self.args.last_action:
    #         inputs = np.hstack((inputs, last_action))
    #     #如果args中设置了reuse_network标志，它将agent_id独热编码向量水平堆叠到inputs数组中。
    #     if self.args.reuse_network:
    #         inputs = np.hstack((inputs, agent_id))
    #     # if self.args.alg == 'qmix':
    #     #从policy的eval_hidden属性中获取当前代理的隐藏状态，使用数组索引。
    #     hidden_state = self.policy.eval_hidden[:, agent_num, :]
    #     # transform the shape of inputs from (42,) to (1,42)
    #     #将inputs和avail_actions数组转换为PyTorch张量，使用unsqueeze(0)在每个张量上添加一个单例维度。
    #     inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    #     avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
    #     if self.args.cuda:
    #         inputs = inputs.cuda()
    #         # if self.args.alg == 'qmix':
    #         hidden_state = hidden_state.cuda()
    #     #将inputs和hidden_state张量传递给policy的eval_rnn方法，获取Q值(q_value)
    #     #还更新policy的eval_hidden属性以反映当前代理的变化。
    #     q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
    #     # choose action from q value 传递Q值、可用动作、epsilon和评估标志，然后返回选择的动作。
    #     #q_value.cpu() 表示将张量 q_value 移动到 CPU 上。
    #     action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
    #     return action
    #
    # def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
    #     """
    #     :param inputs: # q_value of all actions
    #     """
    #     #计算可执行动作的数量，
    #     #avail_actions.sum(dim=1, keepdim=True) 对每个样本计算可执行动作的总数，
    #     # .float() 将结果转换为浮点数，然后使用 .repeat(1, avail_actions.shape[-1]) 在最后一个维度上重复，以匹配 avail_actions 的形状。
    #     action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])
    #     # num of avail_actions
    #     # 先将Actor网络的输出通过softmax转换成概率分布,dim=-1 表示在最后一个维度上进行 softmax 操作。
    #     prob = torch.nn.functional.softmax(inputs, dim=-1)
    #     # add noise of epsilon 引入探索噪音，通过在概率分布上加入噪音，即在原始概率上加权混合一个平均概率。
    #     prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
    #     prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
    #
    #     """
    #     不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
    #     会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
    #     """
    #     #如果 epsilon 为零且处于评估模式时，选择具有最大概率的动作
    #     if epsilon == 0 and evaluate:
    #         action = torch.argmax(prob)#返回概率最大的动作的索引。
    #     else:
    #         #使用torch.distributions.Categorical 从概率分布中采样一个动作。
    #         action = Categorical(prob).sample().long()
    #     return action
#
#     def _get_max_episode_len(self, batch):
#         #从批次中获取包含终止信息的张量，表示每个 episode 是否结束。
#         terminated = batch['terminated']
#         #计算批次中的 episode 数量。
#         episode_num = terminated.shape[0]
#         #获取批次中 o（观察）的张量的第二个维度的大小，即每个 episode 的最大长度。
#         max_episode_len = batch['o'].shape[1]
#         for episode_idx in range(episode_num):
#             for transition_idx in range(self.args.episode_limit):#遍历每个 episode 的每个 transition。
#                 if terminated[episode_idx, transition_idx, 0] == 1:
#                     #如果当前 episode 的长度（transition_idx + 1）大于等于之前记录的最大长度，
#                     #更新 max_episode_len 为当前 episode 的长度。
#                     if transition_idx + 1 >= max_episode_len:
#                         max_episode_len = transition_idx + 1
#                     #一旦发现当前 episode 已经终止，跳出内层循环，继续下一个 episode 的检查。
#                     break
#         # print('1 ',max_episode_len)
#         return max_episode_len
#
    # def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
    #
    #     # different episode has different length, so we need to get max length of the batch
    #     # if on_batch:
    #     #     max_episode_len = max(self._get_max_episode_len(batch), self._get_max_episode_len(on_batch))
    #     # else:
    #     #调用之前定义的 _get_max_episode_len 方法获取当前批次中所有 episode 的最大长度。
    #     max_episode_len = self._get_max_episode_len(batch)
    #     #遍历批次中的所有键（key 是字典 batch 中的键)
    #     for key in batch.keys():
    #         # print(batch[key].shape)
    #         # if on_batch:
    #         #     on_batch[key] = on_batch[key][:, :max_episode_len]
    #         #截取每个键对应的值的第二个维度，将其限制为最大 episode 长度。
    #         batch[key] = batch[key][:, :max_episode_len]
    #
    #     # if self.args.alg == 'dop':
    #     # if on_batch:
    #     #     self.policy.train_critic(on_batch, max_episode_len, train_step, epsilon, best_batch=batch)
    #     # else:
    #     self.policy.learn(batch, max_episode_len, train_step, epsilon)
    #     # else:
    #     #     self.policy.learn(batch, max_episode_len, train_step, epsilon)
    #     #如果当前训练步数大于零且能够整除 self.args.save_cycle，则执行保存模型的操作。
    #     #通过调用 get_model_idx 获取模型的索引，然后调用 save_model 方法保存模型
    #     if train_step > 0 and train_step % self.args.save_cycle == 0:
    #         idx = self.policy.get_model_idx()
    #         self.policy.save_model(idx)
    #         # if self.args.env == 'flight':
    #         #     self.env.save_prob_map(idx)
