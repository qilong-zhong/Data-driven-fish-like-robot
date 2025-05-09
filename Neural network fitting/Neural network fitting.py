import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

EPOCHS = 2500
PLOT_EVERY = 100
STRIDE = 80
length = 0.05
LR = 0.001
DT = 1


def process(data):
    temp = np.array(data['x'].ravel())
    x_orig = temp - temp[0]
    temp = np.array(data['y'].ravel())
    y_orig = temp - temp[0]
    theta_orig = np.array(data['theta'].ravel())
    x = x_orig[::STRIDE]
    y = y_orig[::STRIDE]
    dx = np.diff(x)
    dy = np.diff(y)
    v = np.insert(np.sqrt(dy ** 2 + dx ** 2), 0, 0).reshape((-1, 1))
    tht = np.arctan2(dy, dx)
    dtht = np.diff(tht)
    dtht[dtht >= np.pi] -= 2 * np.pi
    dtht[dtht <= -np.pi] += 2 * np.pi
    dtht_1 = tht[0] - theta_orig[0]
    # dtht = np.insert(dtht, [0, 1], [0, dtht_1]).reshape((-1, 1))
    dtht = np.insert(dtht, [0], [0, dtht_1]).reshape((-1, 1))
    return x_orig, y_orig, theta_orig, dtht, v


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


def plot_result(x_orig, y_orig, a, theta_orig, v_out_max, dtht_out_max, v_in, dtht_in, i):
    plt.subplot(3, 3, i)
    plt.plot(x_orig, y_orig, 'b-', label='Orig')
    position_sim = np.zeros((len(a)+1, 2))
    x_sim = 0
    y_sim = 0
    theta_sim = theta_orig[0]
    for i in range((len(a))):
        theta_sim += dtht_in[i + 1]
        x_sim += v_in[i + 1] * np.cos(theta_sim)
        y_sim += v_in[i + 1] * np.sin(theta_sim)
        position_sim[i + 1, :] = [x_sim, y_sim]
    plt.plot(position_sim[:, 0], position_sim[:, 1], 'm-+', label='Sim')

    position_NN = np.zeros((len(a)+1, 2))
    v_dtht_NN = np.zeros((len(a)+1, 2))
    dtht_NN = 0
    v_NN = 0
    tht_NN = theta_orig[0]
    for i in range(len(a)):
        action_NN = a[i][0]
        [v_NN, dtht_NN] = model.forward(torch.Tensor(np.array([v_NN, dtht_NN, action_NN]))).detach().numpy()
        tht_NN += dtht_NN * dtht_out_max
        position_NN[i+1, :] = (position_NN[i, :] +
                               np.array([v_NN * v_out_max * np.cos(tht_NN), v_NN * v_out_max * np.sin(tht_NN)]) * DT)
        v_dtht_NN[i+1, :] = [v_NN * v_out_max, dtht_NN * dtht_out_max]
    plt.plot(position_NN[:, 0], position_NN[:, 1], 'r-+', label='NN')
    plt.xlim([0, 1])
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return v_dtht_NN


def plot_result2(t_orig, v_orig, v_NN, i):
    plt.subplot(3, 3, i)
    plt.plot(t_orig, v_orig, 'b', label='Exp_orig')
    plt.plot(t_orig, v_NN, 'r--', label='NN')
    plt.xlabel('T [s]')
    plt.ylabel('Velo [m/s]')
    plt.ylim([-0.01, 0.15])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_result3(t_orig, dtht_orig, dtht_NN, i):
    plt.subplot(3, 3, i)
    plt.plot(t_orig, dtht_orig, 'b', label='Exp_orig')
    plt.plot(t_orig, dtht_NN, 'r--', label='NN')
    plt.xlabel('T [s]')
    plt.ylabel('Velo [°/s]')
    plt.ylim([-0.55, 0.55])
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():

    datal = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/l3.csv')  # 2_r
    [x_orig_l, y_orig_l, theta_orig_l, dtht_l, v_l] = process(datal)
    a_l = np.zeros((len(v_l)-1, 1))

    datar = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/r3.csv')  # 2_r
    [x_orig_r, y_orig_r, theta_orig_r, dtht_r, v_r] = process(datar)
    a_r = 2 * np.ones((len(v_r)-1, 1))

    datas = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/s3.csv')  # 2_r
    [x_orig_s, y_orig_s, theta_orig_s, dtht_s, v_s] = process(datas)
    a_s = np.ones((len(v_s)-1, 1))

    datasl = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/sl4.csv')  # 2_r
    [x_orig_sl, y_orig_sl, theta_orig_sl, dtht_sl, v_sl] = process(datasl)
    a_sl = np.ones((len(v_sl)-1, 1))
    a_sl[-19:, 0] = 0

    datasr = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/sr1.csv')  # 2_r
    [x_orig_sr, y_orig_sr, theta_orig_sr, dtht_sr, v_sr] = process(datasr)
    a_sr = np.ones((len(v_sr)-1, 1))
    a_sr[-17:, 0] = 2

    datals = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/ls3.csv')  # 2_r
    [x_orig_ls, y_orig_ls, theta_orig_ls, dtht_ls, v_ls] = process(datals)
    a_ls = np.zeros((len(v_ls)-1, 1))
    a_ls[-7:, 0] = 1

    datalr = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/lr3.csv')  # 2_r lr3
    [x_orig_lr, y_orig_lr, theta_orig_lr, dtht_lr, v_lr] = process(datalr)
    a_lr = np.zeros((len(v_lr)-1, 1))
    a_lr[-18:, 0] = 2

    datars = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/rs3.csv')  # 2_r
    [x_orig_rs, y_orig_rs, theta_orig_rs, dtht_rs, v_rs] = process(datars)
    a_rs = 2 * np.ones((len(v_rs)-1, 1))
    a_rs[-7:, 0] = 1

    datarl = pd.read_csv('D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/rl1.csv')  # 2_r
    [x_orig_rl, y_orig_rl, theta_orig_rl, dtht_rl, v_rl] = process(datarl)
    a_rl = 2 * np.ones((len(v_rl)-1, 1))
    a_rl[-19:, 0] = 0

    v_in = np.concatenate((v_l[0:-1], v_r[0:-1], v_s[0:-1], v_sl[0:-1], v_sr[0:-1],
                           v_ls[0:-1], v_lr[0:-1], v_rs[0:-1], v_rl[0:-1]))
    v_out = np.concatenate((v_l[1:], v_r[1:], v_s[1:], v_sl[1:], v_sr[1:],
                            v_ls[1:],  v_lr[1:], v_rs[1:], v_rl[1:]))
    v_out_max = abs(v_out).max()

    dtht_in = np.concatenate((dtht_l[0:-1], dtht_r[0:-1], dtht_s[0:-1], dtht_sl[0:-1], dtht_sr[0:-1],
                              dtht_ls[0:-1], dtht_lr[0:-1], dtht_rs[0:-1], dtht_rl[0:-1]))
    dtht_out = np.concatenate((dtht_l[1:], dtht_r[1:], dtht_s[1:], dtht_sl[1:], dtht_sr[1:],
                               dtht_ls[1:], dtht_lr[1:], dtht_rs[1:], dtht_rl[1:]))
    dtht_out_max = abs(dtht_out).max()
    action_exp = np.concatenate((a_l, a_r, a_s, a_sl, a_sr, a_ls, a_lr, a_rs, a_rl))
    neural_in = np.concatenate((v_in/v_out_max, dtht_in/dtht_out_max, action_exp.reshape(-1, 1)), axis=1)
    neural_out = np.concatenate((v_out/v_out_max, dtht_out/dtht_out_max), axis=1)
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        y_pred = model(torch.Tensor(neural_in))
        loss = criterion(y_pred, torch.Tensor(neural_out))
        loss.backward()
        optimizer.step()
        if epoch % PLOT_EVERY == 0:
            plt.plot(epoch, np.log10(loss.detach().numpy()), 'b+')
            plt.show()
            plt.pause(1e-5)
    plt.figure(figsize=[7, 7])
    v_dtht_l = plot_result(x_orig_l, y_orig_l, a_l, theta_orig_l, v_out_max, dtht_out_max, v_l, dtht_l, 1)
    v_dtht_r = plot_result(x_orig_r, y_orig_r, a_r, theta_orig_r, v_out_max, dtht_out_max, v_r, dtht_r, 2)
    v_dtht_s = plot_result(x_orig_s, y_orig_s, a_s, theta_orig_s, v_out_max, dtht_out_max, v_s, dtht_s, 3)
    v_dtht_sl = plot_result(x_orig_sl, y_orig_sl, a_sl, theta_orig_sl, v_out_max, dtht_out_max, v_sl, dtht_sl, 4)
    v_dtht_sr = plot_result(x_orig_sr, y_orig_sr, a_sr, theta_orig_sr, v_out_max, dtht_out_max, v_sr, dtht_sr, 5)
    v_dtht_ls = plot_result(x_orig_ls, y_orig_ls, a_ls, theta_orig_ls, v_out_max, dtht_out_max, v_ls, dtht_ls, 6)
    v_dtht_lr = plot_result(x_orig_lr, y_orig_lr, a_lr, theta_orig_lr, v_out_max, dtht_out_max, v_lr, dtht_lr, 7)
    v_dtht_rs = plot_result(x_orig_rs, y_orig_rs, a_rs, theta_orig_rs, v_out_max, dtht_out_max, v_rs, dtht_rs, 8)
    v_dtht_rl = plot_result(x_orig_rl, y_orig_rl, a_rl, theta_orig_rl, v_out_max, dtht_out_max, v_rl, dtht_rl, 9)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.figure(figsize=[7, 7])
    plot_result2(np.arange(len(v_l)), v_l, v_dtht_l[:, 0], 1)
    plot_result2(np.arange(len(v_r)), v_r, v_dtht_r[:, 0],  2)
    plot_result2(np.arange(len(v_s)), v_s, v_dtht_s[:, 0], 3)
    plot_result2(np.arange(len(v_sl)), v_sl, v_dtht_sl[:, 0], 4)
    plot_result2(np.arange(len(v_sr)), v_sr, v_dtht_sr[:, 0], 5)
    plot_result2(np.arange(len(v_ls)), v_ls, v_dtht_ls[:, 0], 6)
    plot_result2(np.arange(len(v_lr)), v_lr, v_dtht_lr[:, 0], 7)
    plot_result2(np.arange(len(v_rs)), v_rs, v_dtht_rs[:, 0], 8)
    plot_result2(np.arange(len(v_rl)), v_rl, v_dtht_rl[:, 0], 9)
    plt.figure(figsize=[7, 7])
    plot_result3(np.arange(len(v_l)), dtht_l, v_dtht_l[:, 1], 1)
    plot_result3(np.arange(len(v_r)), dtht_r, v_dtht_r[:, 1], 2)
    plot_result3(np.arange(len(v_s)), dtht_s, v_dtht_s[:, 1], 3)
    plot_result3(np.arange(len(v_sl)), dtht_sl, v_dtht_sl[:, 1], 4)
    plot_result3(np.arange(len(v_sr)), dtht_sr, v_dtht_sr[:, 1], 5)
    plot_result3(np.arange(len(v_ls)), dtht_ls, v_dtht_ls[:, 1], 6)
    plot_result3(np.arange(len(v_lr)), dtht_lr, v_dtht_lr[:, 1], 7)
    plot_result3(np.arange(len(v_rs)), dtht_rs, v_dtht_rs[:, 1], 8)
    plot_result3(np.arange(len(v_rl)), dtht_rl, v_dtht_rl[:, 1], 9)


main()

torch.save(model.state_dict(), 'D:/gr/rj/pc/Project/test/Xie-fish/data/12.25/code/model_parameters7')
