# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as f
#
#
# class RNN(nn.Module):
#     def __init__(self, input_shape, args):
#         super(RNN, self).__init__()
#         self.args = args
#         self.input_shape = input_shape
#         # 如果配置中启用了卷积（args.conv 为真），则创建卷积层
#         if args.conv:
#             self.conv_size = int((args.map_size - args.kernel_size_1) / args.stride_1 + 1)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(1, args.dim_1, args.kernel_size_1, args.stride_1),  # (1, 50, 50) --> (4, 24, 24)
#                 nn.ReLU(),
#                 nn.Conv2d(args.dim_1, args.dim_2, args.kernel_size_2, args.stride_2, args.padding_2),
#                 # (4, 24, 24) --> (8, 24, 24)
#                 nn.ReLU()
#                 #
#             )
#             self.linear = nn.Linear(args.dim_2 * self.conv_size ** 2, args.conv_out_dim)
#             # input_shape += args.conv_out_dim
#         #print('input shape: ', input_shape)
#         # 定s义一个线性层，将输入形状映射到维度为 args.rnn_hidden_dim-64
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         # 定义一个 GRU 单元，输入和隐藏状态的维度均为 args.rnn_hidden_dim-64
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         # 定义一个线性层和激活函数的序列，最终输出维度为 args.n_actions
#         self.fc2 = nn.Sequential(
#             nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
#             # nn.ReLU(),
#             # nn.Linear(args.rnn_hidden_dim, args.n_actions)
#         )
#
#     def forward(self, obs, hidden_state):
#         if self.args.conv:
#             prob = obs[:, :self.args.map_size ** 2].reshape(-1, 1, self.args.map_size, self.args.map_size)
#             sa = obs[:, self.args.map_size ** 2:]
#             prob_conv = self.conv(prob).reshape(-1, self.args.dim_2 * self.conv_size ** 2)
#             # print(prob_conv.shape)
#             prob_conv = self.linear(prob_conv)
#             # print(prob_conv.shape)
#             # print('prob sa ', prob_conv.shape, sa.shape)
#             obs = torch.cat([prob_conv, sa], 1)
#             # print(self.conv_size)
#             # print('obs ', obs.shape, self.input_shape)
#         # 通过第一个线性层和 ReLU 激活函数处理输入。
#         #print(f"obs.shape: {obs.shape}")
#         x = f.relu(self.fc1(obs))
#         # 将隐藏状态形状重塑为二维矩阵。
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         # 通过 GRU 单元更新隐藏状态。
#         h = self.rnn(x, h_in)
#         # 通过第二个线性层处理隐藏状态，得到最终输出 q。
#         q = self.fc2(h)
#         return q, h
#
# class DuelingDQN(nn.Module):
#     def __init__(self, input_shape, args):
#         super(DuelingDQN, self).__init__()
#         self.args = args
#         self.rnn = RNN(input_shape, args)
#
#         # 为状态值和优势函数分别定义线性层
#         self.V = nn.Linear(args.rnn_hidden_dim, 1)
#         self.A = nn.Linear(args.rnn_hidden_dim, args.n_actions)
#
#     def forward(self, obs, hidden_state):
#         # 使用RNN处理输入和隐藏状态
#         features, h = self.rnn(obs, hidden_state)
#
#         # 分别计算状态值和优势函数
#         values = self.V(features)
#         advantages = self.A(features)
#
#         # 计算Q值：V(s) + A(s,a) - mean(A(s,a))
#         qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
#
#         return qvals, h
#
# class D3QN:
#     def __init__(self, args):
#         self.args = args
#         self.n_actions = args.n_actions  # 动作的数量
#         self.n_agents = args.n_agents  # agent的数量
#         self.state_shape = args.state_shape  # 状态的形状
#         self.obs_shape = args.obs_shape
#         self.seed = args.seed  # 随机种子
#         self.tau = args.tau
#         input_shape = self.obs_shape
#
#         # 根据参数决定RNN的输入维度
#         #如果 args.last_action 为真，表示考虑上一个动作，就将动作数量加到输入维度中。
#         if args.last_action:
#             input_shape += self.n_actions
#         #将 agent 的数量加到输入维度中
#         if args.reuse_network:
#             input_shape += self.n_agents
#
#         # 设置随机种子
#         torch.manual_seed(self.seed)
#         torch.cuda.manual_seed(self.seed)
#         torch.cuda.manual_seed_all(self.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#         # 初始化网络
#         self.eval_ddqn = DuelingDQN(input_shape, args)
#         self.target_ddqn = DuelingDQN(input_shape, args)
#         self.args = args
#         if self.args.cuda:
#             self.eval_ddqn.cuda()
#             self.target_ddqn.cuda()
#
#         #设置保存模型的目录，其中包含了环境名称、随机种子、算法名称、以及其他一些配置。
#         self.model_dir = './model/'+ args.env + '_Seed' + str(
#             args.seed) + '_D3QN' + '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
#                                                                      args.target_mode)
#
#         # 如果存在模型则加载模型
#         if self.args.load_model:
#             model_index = self.get_model_idx() - 1
#             if os.path.exists(os.path.join(self.model_dir, str(model_index) + '_ddqn_net_params.pkl')):
#                 path_ddqn = os.path.join(self.model_dir, str(model_index) + '_ddqn_net_params.pkl')
#                 path_target = os.path.join(self.model_dir, str(model_index) + '_target_ddqn_net_params.pkl')
#                 map_location = 'cuda' if self.args.cuda else 'cpu'
#                 self.eval_ddqn.load_state_dict(torch.load(path_ddqn, map_location=map_location))
#                 self.eval_target.load_state_dict(torch.load(path_target, map_location=map_location))
#                 print('Successfully load the model: {}'.format(path_ddqn))
#             else:
#                 raise Exception("No model!")
#
#         # update target net params
#         self.target_ddqn.load_state_dict(self.eval_target.state_dict())
#
#         #获取DDQN模型的参数列表。
#         self.ddqn_parameters = list(self.eval_ddqn.parameters())
#
#         # 设置优化器
#         if args.optimizer == "RMS":
#             self.optimizer = torch.optim.RMSprop(self.ddqn_parameters, lr=args.lr)
#         elif args.optimizer == "Adam":
#             self.optimizer = torch.optim.Adam(self.ddqn_parameters, lr=args.lr)
#         else:
#             raise Exception('No such optimizer')
#
#         # 初始化隐藏状态
#         self.eval_hidden = None
#         self.target_hidden = None
#         print('Init alg D3QN(seed = {})'.format(self.seed))
#
#     def init_hidden(self, episode_num):
#         self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
#         self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
#
#     def soft_update(self):
#         print('update')
#         for param, target_param in zip(self.eval_ddqn.parameters(), self.target_ddqn.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#     def _get_inputs(self, batch, transition_idx):
#         # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
#         obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
#                                   batch['o_next'][:, transition_idx], batch['u_onehot'][:]
#         #获取批次中的 episode 数量。
#         episode_num = obs.shape[0]
#         inputs, inputs_next = [], []
#         #将观测添加到输入列表中。
#         inputs.append(obs)
#         inputs_next.append(obs_next)
#         #给inputs添加上一个动作、agent编号
#         if self.args.last_action:#如果模型考虑了上一个动作的影响，将上一个动作的独热编码添加到输入列表中。
#             if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
#                 inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
#             else:
#                 inputs.append(u_onehot[:, transition_idx - 1])
#             inputs_next.append(u_onehot[:, transition_idx])
#         if self.args.reuse_network:
#             # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
#             # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
#             # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
#             inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
#             inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
#         # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
#         # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
#         inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
#         inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
#         return inputs, inputs_next
#
#     def learn(self,batch, max_episode_len, train_step, epsilon):
#         #获取批次中的episode数量。
#         episode_num = batch['o'].shape[0]
#         #初始化隐藏状态，确保每个episode的每个agent都有一个对应的隐藏状态。
#         self.init_hidden(episode_num)
#
#         ##遍历批次中的所有键（key 是字典 batch 中的键)
#         for key in batch.keys():  # 把batch里的数据转化成tensor
#             if key == 'u':
#                 batch[key] = torch.tensor(batch[key], dtype=torch.long)  # 将其转换为torch.long类型。
#             else:
#                 batch[key] = torch.tensor(batch[key], dtype=torch.float32)  # 其他键，将其转换为torch.float32类型。
#         # 从batch中提取数据
#         o = batch['o']
#         u = batch['u']
#         r = batch['r']
#         o_next = batch['o_next']
#         terminated = batch['terminated']
#         #获取可行动作的掩码，形状为 (episode_num, max_episode_len, n_agents, n_actions)
#         avail_actions = batch['avail_u']
#
#         if self.args.cuda:
#             o = o.cuda()
#             u = u.cuda()
#             r = r.cuda()
#             o_next = o_next.cuda()
#             terminated = terminated.cuda()
#
#         # 在此处添加调试代码以检查 'o' 的形状和数据类型
#         #print("Shape of o:", batch['o'].shape)
#         #print("Data type of o:", batch['o'].dtype)
#
#         # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
#         q_evals, q_targets = self.get_q_values(batch, max_episode_len)
#         # print('q_evals_shape：', q_evals.shape)
#         # print('q_evals：', q_evals)
#         # print('q_targets_shape：', q_targets.shape)
#         # print('q_targets：', q_targets)
#         # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
#         q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
#
#         # 取每个agent动作对应的Q+1值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
#         q_targets = torch.gather(q_targets, dim=3, index=u).squeeze(3)
#         # print('q_evals_shape：', q_evals.shape)
#         # print('q_evals：', q_evals)
#         # print('q_targets_shape：', q_targets.shape)
#         # print('q_targets：', q_targets)
#         #     #print('current_q_values_shape', current_q_values.shape)
#         #     #print('current_q_values:', current_q_values)
#         #     current_q = current_q_values.max(dim=1)[0]#选择出每个智能体q最大的值
#         #     #print('current_q_shape', current_q.shape)
#         #     #print('current_q:', current_q)
#         # print("r shape:", r.shape)
#         # print("r:",r)
#         targets = r + self.args.gamma * q_targets * (1 - terminated)
#
#         loss = f.mse_loss(q_evals, targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
#         #      self.target_ddqn.load_state_dict(self.eval_ddqn.state_dict())
#         self.soft_update()
#
#     def get_q_values(self, batch, max_episode_len):
#         episode_num = batch['o'].shape[0]#从输入的数据批次中提取出episode的数量
#         q_evals, q_targets = [], []
#         for transition_idx in range(max_episode_len):
#             #获取当前和下一时刻的输入
#             inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
#             if self.args.cuda:
#                 inputs = inputs.cuda()
#                 inputs_next = inputs_next.cuda()
#                 self.eval_hidden = self.eval_hidden.cuda()
#                 #self.target_hidden = self.target_hidden.cuda()
#             #计算当前时刻的Q值（评估Q值）和下一时刻的Q值（目标Q值），同时更新网络的隐藏状态
#             q_eval, self.eval_hidden = self.eval_ddqn(inputs, self.eval_hidden)#得到的q_eval维度为(episode_num*n_agents, n_actions)
#             q_target, self.target_hidden = self.target_ddqn(inputs_next, self.target_hidden)
#
#             # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
#             q_eval = q_eval.view(episode_num, self.n_agents, -1)
#             q_target = q_target.view(episode_num, self.n_agents, -1)
#             q_evals.append(q_eval)
#             q_targets.append(q_target)
#         # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
#         # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
#         q_evals = torch.stack(q_evals, dim=1)
#         q_targets = torch.stack(q_targets, dim=1)
#         return q_evals, q_targets
#
#
#     def save_model(self, train_step):
#         #如果模型保存目录不存在，则创建。
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#         print('Model saved')
#         # 获取当前模型的索引
#         idx = str(self.get_model_idx())
#         #将 RNN 网络的参数保存到文件中。
#         torch.save(self.eval_ddqn.state_dict(), self.model_dir + '/' + idx + '_ddqn_net_params.pkl')
#         torch.save(self.eval_target.state_dict(), self.model_dir + '/' + idx + '_target_ddqn_net_params.pkl')
#
#     def get_model_idx(self):
#         #如果模型保存目录不存在，则创建。
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#             return 0
#         idx = 0
#         #获取模型目录下的所有文件。
#         models = os.listdir(self.model_dir)
#         #遍历所有文件。
#         for model in models:
#             #从文件名中提取数字。
#             num = int(model.split('_')[0])
#             #更新索引，确保新的索引比已存在的索引大。
#             idx = max(idx, num)
#         idx += 1
#         return idx
#
#     def load_model(self, d3qn_root):
#         self.eval_ddqn.load_state_dict(torch.load(d3qn_root))
#         self.target_ddqn.load_state_dict(self.eval_target.state_dict())


import os
import torch
import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        # 如果配置中启用了卷积（args.conv 为真），则创建卷积层
        if args.conv:
            self.conv_size = int((args.map_size - args.kernel_size_1) / args.stride_1 + 1)#24
            self.conv = nn.Sequential(
                #1,4,4,2
                nn.Conv2d(1, args.dim_1, args.kernel_size_1, args.stride_1),  # (1, 50, 50) --> (4, 24, 24)
                nn.ReLU(),#4,1,2,1,1
                nn.Conv2d(args.dim_1, args.dim_2, args.kernel_size_2, args.stride_2, args.padding_2),
                # (4, 24, 24) --> (8, 24, 24)
                nn.ReLU()
                #
            )
            self.linear = nn.Linear(args.dim_2 * self.conv_size ** 2, args.conv_out_dim)
            # input_shape += args.conv_out_dim
        #print('input shape: ', input_shape)
        # 定s义一个线性层，将输入形状映射到维度为 args.rnn_hidden_dim-64
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # 定义一个 GRU 单元，输入和隐藏状态的维度均为 args.rnn_hidden_dim-64
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # 定义一个线性层和激活函数的序列，最终输出维度为 args.n_actions
        self.fc2 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            # nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

    def forward(self, obs, hidden_state):
        if self.args.conv:
            prob = obs[:, :self.args.map_size ** 2].reshape(-1, 1, self.args.map_size, self.args.map_size)
            sa = obs[:, self.args.map_size ** 2:]
            prob_conv = self.conv(prob).reshape(-1, self.args.dim_2 * self.conv_size ** 2)
            # print(prob_conv.shape)
            prob_conv = self.linear(prob_conv)
            # print(prob_conv.shape)
            # print('prob sa ', prob_conv.shape, sa.shape)
            obs = torch.cat([prob_conv, sa], 1)
            # print(self.conv_size)
            # print('obs ', obs.shape, self.input_shape)
        # 通过第一个线性层和 ReLU 激活函数处理输入。
        #print(f"obs.shape: {obs.shape}")
        x = f.relu(self.fc1(obs))
        # 将隐藏状态形状重塑为二维矩阵。
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # 通过 GRU 单元更新隐藏状态。
        h = self.rnn(x, h_in)
        # 通过第二个线性层处理隐藏状态，得到最终输出 q。
        q = self.fc2(h)
        return q, h

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, args):
        super(DuelingDQN, self).__init__()
        self.args = args
        self.rnn = RNN(input_shape, args)

        # 为状态值和优势函数分别定义线性层
        self.V = nn.Linear(args.rnn_hidden_dim, 1)
        self.A = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        # 使用RNN处理输入和隐藏状态
        features, h = self.rnn(obs, hidden_state)

        # 分别计算状态值和优势函数
        values = self.V(features)
        advantages = self.A(features)

        # 计算Q值：V(s) + A(s,a) - mean(A(s,a))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return qvals, h

class D3QN:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions  # 动作的数量
        self.n_agents = args.n_agents  # agent的数量
        self.state_shape = args.state_shape  # 状态的形状
        self.obs_shape = args.obs_shape
        self.seed = args.seed  # 随机种子
        self.tau = args.tau
        input_shape = self.obs_shape

        # 根据参数决定RNN的输入维度
        #如果 args.last_action 为真，表示考虑上一个动作，就将动作数量加到输入维度中。
        if args.last_action:
            input_shape += self.n_actions
        #将 agent 的数量加到输入维度中
        if args.reuse_network:
            input_shape += self.n_agents

        # 设置随机种子
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 初始化网络
        self.eval_ddqn = DuelingDQN(input_shape, args)
        self.target_ddqn = DuelingDQN(input_shape, args)
        self.args = args
        if self.args.cuda:
            self.eval_ddqn.cuda()
            self.target_ddqn.cuda()

        #设置保存模型的目录，其中包含了环境名称、随机种子、算法名称、以及其他一些配置。
        self.model_dir = './model/'+ args.env + '_Seed' + str(
            args.seed) + '_D3QN' + '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode,
                                                                     args.target_mode)

        # 如果存在模型则加载模型
        if self.args.load_model:
            model_index = self.get_model_idx() - 1
            if os.path.exists(os.path.join(self.model_dir, str(model_index) + '_ddqn_net_params.pkl')):
                path_ddqn = os.path.join(self.model_dir, str(model_index) + '_ddqn_net_params.pkl')
                map_location = 'cuda' if self.args.cuda else 'cpu'
                self.eval_ddqn.load_state_dict(torch.load(path_ddqn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_ddqn))
            else:
                raise Exception("No model!")

        # update target net params
        self.target_ddqn.load_state_dict(self.eval_ddqn.state_dict())

        #获取DDQN模型的参数列表。
        self.ddqn_parameters = list(self.eval_ddqn.parameters())

        # 设置优化器
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.ddqn_parameters, lr=args.lr)
        elif args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.ddqn_parameters, lr=args.lr)
        else:
            raise Exception('No such optimizer')

        # 初始化隐藏状态
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg D3QN(seed = {})'.format(self.seed))

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def soft_update(self):
        print('update')
        for param, target_param in zip(self.eval_ddqn.parameters(), self.target_ddqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        #获取批次中的 episode 数量。
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        #将观测添加到输入列表中。
        inputs.append(obs)
        inputs_next.append(obs_next)
        #给inputs添加上一个动作、agent编号
        if self.args.last_action:#如果模型考虑了上一个动作的影响，将上一个动作的独热编码添加到输入列表中。
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def learn(self,batch, max_episode_len, train_step, epsilon):
        #获取批次中的episode数量。
        episode_num = batch['o'].shape[0]
        #初始化隐藏状态，确保每个episode的每个agent都有一个对应的隐藏状态。
        self.init_hidden(episode_num)

        ##遍历批次中的所有键（key 是字典 batch 中的键)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key].astype(float), dtype=torch.long) # 将其转换为torch.long类型。
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)  # 其他键，将其转换为torch.float32类型。
        # 从batch中提取数据
        o = batch['o']
        u = batch['u']
        r = batch['r']
        o_next = batch['o_next']
        terminated = batch['terminated']
        #获取可行动作的掩码，形状为 (episode_num, max_episode_len, n_agents, n_actions)
        avail_actions = batch['avail_u']

        if self.args.cuda:
            o = o.cuda()
            u = u.cuda()
            r = r.cuda()
            o_next = o_next.cuda()
            terminated = terminated.cuda()
            avail_actions = avail_actions.cuda()

        # 在此处添加调试代码以检查 'o' 的形状和数据类型
        #print("Shape of o:", batch['o'].shape)
        #print("Data type of o:", batch['o'].dtype)

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        # print('q_evals_shape：', q_evals.shape)
        # print('q_evals：', q_evals)
        # print('q_targets_shape：', q_targets.shape)
        # print('q_targets：', q_targets)
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 取每个agent动作对应的Q+1值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_targets = torch.gather(q_targets, dim=3, index=u).squeeze(3)
        # print('q_evals_shape：', q_evals.shape)
        # print('q_evals：', q_evals)
        # print('q_targets_shape：', q_targets.shape)
        # print('q_targets：', q_targets)
        #     #print('current_q_values_shape', current_q_values.shape)
        #     #print('current_q_values:', current_q_values)
        #     current_q = current_q_values.max(dim=1)[0]#选择出每个智能体q最大的值
        #     #print('current_q_shape', current_q.shape)
        #     #print('current_q:', current_q)
        r=r.squeeze(2)
        targets = r + self.args.gamma * q_targets * (1 - terminated)
        # print('targets_shape：', targets.shape)
        # print('targets：', targets)

        loss = f.mse_loss(q_evals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
        #      self.target_ddqn.load_state_dict(self.eval_ddqn.state_dict())
        self.soft_update()

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]#从输入的数据批次中提取出episode的数量
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            #获取当前和下一时刻的输入
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                #self.target_hidden = self.target_hidden.cuda()
            #计算当前时刻的Q值（评估Q值）和下一时刻的Q值（目标Q值），同时更新网络的隐藏状态
            q_eval, self.eval_hidden = self.eval_ddqn(inputs, self.eval_hidden)#得到的q_eval维度为(episode_num*n_agents, n_actions)
            q_target, self.target_hidden = self.target_ddqn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets


    def save_model(self, train_step):
        #如果模型保存目录不存在，则创建。
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print('Model saved')
        # 获取当前模型的索引
        idx = str(self.get_model_idx())
        #将 RNN 网络的参数保存到文件中。
        torch.save(self.eval_ddqn.state_dict(), self.model_dir + '/' + idx + '_ddqn_net_params.pkl')

    def get_model_idx(self):
        #如果模型保存目录不存在，则创建。
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            return 0
        idx = 0
        #获取模型目录下的所有文件。
        models = os.listdir(self.model_dir)
        #遍历所有文件。
        for model in models:
            #从文件名中提取数字。
            num = int(model.split('_')[0])
            #更新索引，确保新的索引比已存在的索引大。
            idx = max(idx, num)
        idx += 1
        return idx

    def load_model(self, d3qn_root):
        self.eval_ddqn.load_state_dict(torch.load(d3qn_root))
        self.target_ddqn.load_state_dict(self.eval_ddqn.state_dict())
