import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils


# physical dimensions
fish_len = 0.05
steps = 100
r_range_s = 0
r_range_l = 1.2
phi_range = np.pi
eta_range = np.pi
r_conv = 0.1  # 0.06
phi_conv = 0.3  # 0.25
target_point = [0, 0, 0]
# RL hyper parameter
actor_lr = 1e-3
critic_lr = 1e-3
episodes = 30000
hidden_dim = 128
gamma = 0.94
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dim = 3
action_dim = 3

EPOCHS = 2000
STRIDE = 80
DT = 1
LR = 0.001


def process(data):
    temp = np.array(data['x'].ravel())
    x_orig = temp - temp[0]
    temp = np.array(data['y'].ravel())
    y_orig = temp - temp[0]
    theta_orig = np.array(data['theta'].ravel())
    x_ = x_orig[::STRIDE]
    y_ = y_orig[::STRIDE]
    dx = np.diff(x_)
    dy = np.diff(y_)
    v = np.insert(np.sqrt(dy ** 2 + dx ** 2), 0, 0).reshape((-1, 1))
    tht = np.arctan2(dy, dx)
    dtht = np.diff(tht)
    dtht[dtht >= np.pi] -= 2 * np.pi
    dtht[dtht <= -np.pi] += 2 * np.pi
    dtht_1 = tht[0] - theta_orig[0]
    dtht = np.insert(dtht, [0], [0, dtht_1]).reshape((-1, 1))
    return x_orig, y_orig, theta_orig, dtht, v


def plot_result(x_orig, y_orig, a, tht, v_out_max_, dtht_out_max_, v_in_, dtht_in_, i_):
    plt.subplot(3, 3, i_)
    plt.plot(x_orig, y_orig, 'b-', label='Right')
    position_sim = np.zeros((len(a), 2))
    x_sim = 0
    y_sim = 0
    theta_sim = tht[0]
    for i_ in range((len(a) - 1)):
        x_sim += v_in_[i_ + 1] * np.cos(theta_sim)
        y_sim += v_in_[i_ + 1] * np.sin(theta_sim)
        theta_sim += dtht_in_[i_ + 1]
        position_sim[i_ + 1, :] = [x_sim, y_sim]
    plt.plot(position_sim[:, 0], position_sim[:, 1], 'm-+', label='Right')
    position = np.zeros((len(a), 2))
    v_dtht = np.zeros((len(a), 2))
    dtht = 0
    v = 0
    tht = tht[0]
    for i_ in range(len(a)-1):
        action_ = a[i_][0]
        [v, dtht] = model.forward(torch.Tensor(np.array([v, dtht, action_]))).detach().numpy()
        position[i_+1, :] = position[i_, :] + np.array([v*v_out_max_*np.cos(tht), v*v_out_max_*np.sin(tht)]) * DT
        tht += dtht * dtht_out_max_
        v_dtht[i_+1, :] = [v*v_out_max_, dtht * dtht_out_max_]
    plt.plot(position[:, 0], position[:, 1], 'r-+', label='NN')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return v_dtht


def plot_result2(t_orig_v, v_orig, t_v, v_NN, i_):
    plt.subplot(3, 3, i_)
    plt.plot(t_orig_v, v_orig, 'b', label='Exp_orig')
    plt.plot(t_v, v_NN, 'r--', label='NN')
    plt.xlabel('T [s]')
    plt.ylabel('Velo [m/s]')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_result3(t_orig_dtht, dtht_orig, t_dtht, dtht_NN, i_):
    plt.subplot(3, 3, i_)
    plt.plot(t_orig_dtht, dtht_orig, 'b', label='Exp_orig')
    plt.plot(t_dtht, dtht_NN, 'r--', label='NN')
    plt.xlabel('T [s]')
    plt.ylabel('Velo [°/s]')
    plt.legend()
    plt.tight_layout()
    plt.show()


datal = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/l3.csv')
[x_orig_l, y_orig_l, theta_orig_l, dtht_l, v_l] = process(datal)
a_l = np.zeros((len(v_l)-1, 1))

datar = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/r3.csv')
[x_orig_r, y_orig_r, theta_orig_r, dtht_r, v_r] = process(datar)
a_r = 2 * np.ones((len(v_r)-1, 1))

datas = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/s3.csv')
[x_orig_s, y_orig_s, theta_orig_s, dtht_s, v_s] = process(datas)
a_s = np.ones((len(v_s)-1, 1))

datasl = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/sl4.csv')
[x_orig_sl, y_orig_sl, theta_orig_sl, dtht_sl, v_sl] = process(datasl)
a_sl = np.ones((len(v_sl)-1, 1))
a_sl[-19:, 0] = 0

datasr = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/sr1.csv')
[x_orig_sr, y_orig_sr, theta_orig_sr, dtht_sr, v_sr] = process(datasr)
a_sr = np.ones((len(v_sr)-1, 1))
a_sr[-17:, 0] = 2

datals = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/ls3.csv')
[x_orig_ls, y_orig_ls, theta_orig_ls, dtht_ls, v_ls] = process(datals)
a_ls = np.zeros((len(v_ls)-1, 1))
a_ls[-7:, 0] = 1

datalr = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/lr3.csv')
[x_orig_lr, y_orig_lr, theta_orig_lr, dtht_lr, v_lr] = process(datalr)
a_lr = np.zeros((len(v_lr)-1, 1))
a_lr[-18:, 0] = 2

datars = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/rs3.csv')
[x_orig_rs, y_orig_rs, theta_orig_rs, dtht_rs, v_rs] = process(datars)
a_rs = 2 * np.ones((len(v_rs)-1, 1))
a_rs[-7:, 0] = 1

datarl = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/rl1.csv')
[x_orig_rl, y_orig_rl, theta_orig_rl, dtht_rl, v_rl] = process(datarl)
a_rl = 2 * np.ones((len(v_rl)-1, 1))
a_rl[-19:, 0] = 0

v_in = np.concatenate((v_l[0:-1], v_r[0:-1], v_s[0:-1], v_sl[0:-1], v_sr[0:-1],
                       v_ls[0:-1], v_lr[0:-1], v_rs[0:-1], v_rl[0:-1]))
v_out = np.concatenate((v_l[1:], v_r[1:], v_s[1:], v_sl[1:], v_sr[1:],
                        v_ls[1:], v_lr[1:], v_rs[1:], v_rl[1:]))
v_out_max = abs(v_out).max()

dtht_in = np.concatenate((dtht_l[0:-1], dtht_r[0:-1], dtht_s[0:-1], dtht_sl[0:-1], dtht_sr[0:-1],
                          dtht_ls[0:-1], dtht_lr[0:-1], dtht_rs[0:-1], dtht_rl[0:-1]))
dtht_out = np.concatenate((dtht_l[1:], dtht_r[1:], dtht_s[1:], dtht_sl[1:], dtht_sr[1:],
                           dtht_ls[1:], dtht_lr[1:], dtht_rs[1:], dtht_rl[1:]))
dtht_out_max = abs(dtht_out).max()
action_exp = np.concatenate((a_l, a_r, a_s, a_sl, a_sr, a_ls, a_lr, a_rs, a_rl))
neural_in = np.concatenate((v_in / v_out_max, dtht_in / dtht_out_max, action_exp.reshape(-1, 1)), axis=1)
neural_out = np.concatenate((v_out / v_out_max, dtht_out / dtht_out_max), axis=1)


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features*8)
        self.fc2 = nn.Linear(hidden_features*8, hidden_features*4)
        self.fc3 = nn.Linear(hidden_features*4, hidden_features*2)
        self.fc4 = nn.Linear(hidden_features*2, out_features)

    def forward(self, x_):
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.fc3(x_)
        x_ = F.relu(x_)
        x_ = self.fc4(x_)
        return x_


model = Model(3, 64, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.load_state_dict(torch.load('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25 with code/code/model_parameters6'))


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim_, hidden_dim_, action_dim_):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim_, hidden_dim_)
        self.fc2 = torch.nn.Linear(hidden_dim_, action_dim_)

    def forward(self, x_):
        x_ = F.relu(self.fc1(x_))
        return F.softmax(self.fc2(x_), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim_, hidden_dim_):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim_, hidden_dim_)
        self.fc2 = torch.nn.Linear(hidden_dim_, 1)

    def forward(self, x_):
        x_ = F.relu(self.fc1(x_))
        return self.fc2(x_)


class PPO:
    def __init__(self, state_dim_, hidden_dim_, action_dim_, actor_lr_, critic_lr_, lmbda_, epochs_, eps_, gamma_,
                 device_):
        self.actor = PolicyNet(state_dim_, hidden_dim_, action_dim_).to(device_)
        self.critic = ValueNet(state_dim_, hidden_dim_).to(device_)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr_)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr_)
        self.gamma_ = gamma_
        self.lmbda_ = lmbda_
        self.epochs_ = epochs_
        self.eps_ = eps_
        self.device_ = device_

    def take_action(self, state_):
        state_ = torch.tensor([state_], dtype=torch.float).to(self.device_)
        probs = self.actor(state_)
        action_dist = torch.distributions.Categorical(probs)
        action_ = action_dist.sample()
        return action_.item()

    def take_action2(self, state_):
        state_ = torch.tensor([state_], dtype=torch.float).to(self.device_)
        probs = self.actor(state_)
        # 使用argmax找到概率最大的动作索引
        action_ = torch.argmax(probs)
        return action_.item()

    def update(self, transition_dict_):
        states_ = torch.tensor(transition_dict_['states'], dtype=torch.float).to(self.device_)
        actions = torch.tensor(transition_dict_['actions']).view(-1, 1).to(self.device_)
        rewards = torch.tensor(transition_dict_['rewards'], dtype=torch.float).view(-1, 1).to(self.device_)
        next_states = torch.tensor(transition_dict_['next_states'], dtype=torch.float).to(self.device_)
        dones = torch.tensor(transition_dict_['dones'], dtype=torch.float).view(-1, 1).to(self.device_)
        td_target = rewards + self.gamma_ * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states_)
        advantage = rl_utils.compute_advantage(self.gamma_, self.lmbda_, td_delta.cpu()).to(self.device_)
        old_log_probs = torch.log(self.actor(states_).gather(1, actions)).detach()
        for _ in range(self.epochs_):
            log_probs = torch.log(self.actor(states_).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_, 1 + self.eps_) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states_), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
episode_rewards = []
recent_average_rewards_e = []
recent_average_rewards_e50 = []
reward_pre = 0
recent_average_reward_e50 = 0
difficulty_threshold = 0.94  # 难度提升的阈值
current_difficulty = 0  # 初始难度等级
difficulty_levels = {
    0: {'steps': 25, 'r_range_s': 0.3, 'r_range_l': 0.5, 'phi_range': np.pi/6, 'eta_range': np.pi/6},
    1: {'steps': 50, 'r_range_s': 0.5, 'r_range_l': 1.2, 'phi_range': np.pi, 'eta_range': np.pi/2},
    2: {'steps': 60, 'r_range_s': 0, 'r_range_l': 1.2, 'phi_range': np.pi, 'eta_range': np.pi},
}

for episode in tqdm(range(episodes)):
    if episode > 100 and recent_average_reward_e50 >= difficulty_threshold and current_difficulty < max(difficulty_levels.keys()):
        current_difficulty += 1
        print(f"Episode {episode}: Increased difficulty to Level  {current_difficulty}")
    steps = difficulty_levels[current_difficulty]['steps']
    r_range_s = difficulty_levels[current_difficulty]['r_range_s']
    r_range_l = difficulty_levels[current_difficulty]['r_range_l']
    phi_range = difficulty_levels[current_difficulty]['phi_range']
    eta_range = difficulty_levels[current_difficulty]['eta_range']
    r = np.random.uniform(r_range_s, r_range_l)
    phi = np.random.uniform(-phi_range, phi_range)
    eta = np.random.uniform(-eta_range, eta_range)
    if phi >= np.pi:
        phi = phi - 2 * np.pi
    if phi <= -np.pi:
        phi = phi + 2 * np.pi
    if eta >= np.pi:
        eta = eta - 2 * np.pi
    if eta <= -np.pi:
        eta = eta + 2 * np.pi
    state_cartesian = [- r * np.cos(eta), - r * np.sin(eta), phi]
    state = [r, phi, eta]
    episode_reward = 0
    states = []
    coordinates = []
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    D = 0
    dtheta = 0
    success = 0

    for step in range(steps):
        done = False
        action = agent.take_action(state)
        [D, dtheta] = model.forward(torch.Tensor([D, dtheta, action])).detach().numpy()
        phi = phi + dtheta * dtht_out_max
        velo = D * v_out_max
        if phi >= np.pi:
            phi = phi - 2 * np.pi
        if phi <= -np.pi:
            phi = phi + 2 * np.pi
        x = state_cartesian[0] + velo * np.cos(phi) * DT
        y = state_cartesian[1] + velo * np.sin(phi) * DT
        r = np.sqrt(x ** 2 + y ** 2)
        eta = np.arctan2(- y, - x)
        if eta >= np.pi:
            eta = eta - 2 * np.pi
        if eta <= -np.pi:
            eta = eta + 2 * np.pi
        next_state = [r, phi, eta]
        if r < r_conv and abs(phi) < phi_conv:
            reward = 1
            success = 1
            done = True
        else:
            reward = -0.001
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        episode_reward += reward
        state = np.copy(next_state)
        state_cartesian = np.copy([x, y, phi])
        if done or step == (steps - 1):
            if success == 1:
                reward_e = 1
            else:
                reward_e = 0
            recent_average_rewards_e.append(reward_e)
            break
    agent.update(transition_dict)
    episode_rewards.append(episode_reward)
    if episode <= 50:
        recent_average_reward_e50 = sum(recent_average_rewards_e) / (episode + 1)
        recent_average_rewards_e50.append(recent_average_reward_e50)
    else:
        recent_average_reward_e50 = sum(recent_average_rewards_e[-50:]) / 50
        recent_average_rewards_e50.append(recent_average_reward_e50)
    if episode % 100 == 0 and episode > 0:
        mean_reward = np.mean(recent_average_rewards_e[-100:])
        plt.plot([episode-100, episode], [reward_pre, mean_reward], 'b-+')
        reward_pre = mean_reward
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.show()
        plt.pause(1e-5)

    if current_difficulty == max(difficulty_levels.keys()) and recent_average_reward_e50 >= 1:
        print(f"Episode {episode}: Reach the maximum difficulty and average reward of 1, exit the loop.")
        break


plt.figure(figsize=(6, 4))
episodes = episode + 1
plt.plot(range(episodes), recent_average_rewards_e50, 'b', label='recent_average_rewards')
plt.plot([0, episodes], [0, 0], 'k--')
plt.plot([0, episodes], [1, 1], 'k--')
plt.plot([0, episodes], [0.94, 0.94], 'r--')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward for Last 50 Episodes')
plt.show()
