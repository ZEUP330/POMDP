#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
import math
import cv2
import random
import time


#####################  hyper parameters  ####################

LR_A = 0.001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 60
N_STATES = 3
RENDER = False
EPSILON = 0.9
###############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet, self).__init__()
        self.vgg = models.resnet18(True)
        # self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        # self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        # self.pool = nn.MaxPool2d(5, 2, 2)
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 64)
        # self.fc3 = nn.Linear(64 + 512 * 28 * 28, 3)
        self.fc3 = nn.Linear(64 + 2000, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, dep, s):
        # rgb = F.relu(self.pool(self.conv1(rgb)))
        # rgb = F.relu(self.pool(self.conv2(rgb)))
        # rgb = F.relu(self.pool(self.conv3(rgb)))
        # rgb = rgb.reshape((-1, 256 * 28 * 28))
        # dep = F.relu(self.pool(self.conv1(dep)))
        # dep = F.relu(self.pool(self.conv2(dep)))
        # dep = F.relu(self.pool(self.conv3(dep)))
        # dep = dep.reshape((-1, 256 * 28 * 28))
        rgb = self.vgg(rgb)
        dep = self.vgg(dep)
        a = F.relu(self.fc1(s))
        x = F.relu(self.fc2(a))
        x =torch.cat((rgb.float(), dep.float(), x.float()), dim=1)
        x = self.fc3(x)
        return x


class CNet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(CNet,self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        # self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        # self.pool = nn.MaxPool2d(5, 2, 2)
        self.vgg = models.resnet18(True)
        # self.dense121 = models.resnet50(False)  # (1, 1000)
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 64)
        # self.fc3 = nn.Linear(64 + 512 * 28 * 28, 1)
        self.fc3 = nn.Linear(64 + 2000, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, dep, s, a):
        x = torch.cat((s, a), 1)
        # rgb = F.relu(self.pool(self.conv1(rgb)))
        # rgb = F.relu(self.pool(self.conv2(rgb)))
        # rgb = F.relu(self.pool(self.conv3(rgb)))
        # rgb = rgb.reshape((-1, 256 * 28 * 28))
        # dep = F.relu(self.pool(self.conv1(dep)))
        # dep = F.relu(self.pool(self.conv2(dep)))
        # dep = F.relu(self.pool(self.conv3(dep)))
        # dep = dep.reshape((-1, 256 * 28 * 28))
        rgb = self.vgg(rgb)
        dep = self.vgg(dep)
        a = F.relu(self.fc1(x))
        x = F.relu(self.fc2(a))
        x =torch.cat((rgb.float(), dep.float(), x.float()), dim=1)
        x = self.fc3(x)
        return x


class DDPG(object):
    def __init__(self):
        self.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.device = torch.device("cuda:0")
        self.memory = np.zeros((MEMORY_CAPACITY, 3+3+1+3))
        self.memory_counter = 0  # 记忆库计数
        self.Actor_eval = ANet().to(self.device)
        self.Actor_target = ANet().to(self.device)
        self.Critic_eval = CNet().to(self.device)
        self.Critic_target = CNet().to(self.device)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.f = 0
        self.dep = np.load("C:/Users/VCC/Desktop/[0.5908450974804389, 0.0754887624414724, 0.75].npy")
        self.rgb = cv2.imread("C:/Users/VCC/Desktop/[0.5908450974804389, 0.0754887624414724, 0.75].png")
        self.r = self.rgb = np.array(self.rgb).reshape(1, 3, 224, 224)
        for i in range(BATCH_SIZE - 1):
            self.r = np.concatenate((self.r, self.rgb), axis=0)
        self.rgb = torch.FloatTensor(self.r).to(self.device)
        self.dep = self.dep.reshape(1, 1, 224, 224)
        self.d = self.dep = np.concatenate((self.dep, self.dep, self.dep), axis=1)
        for i in range(BATCH_SIZE - 1):
            self.d = np.concatenate((self.d, self.dep), axis=0)
        self.dep = torch.FloatTensor(self.d).to(self.device)

    def choose_action(self, rgb, dep, s):
        state = torch.FloatTensor(np.array(s).reshape(1, -1)).to(self.device)
        rgb = torch.FloatTensor(np.array(rgb).reshape(1, 3, 224, 224)).to(self.device)
        dep = torch.FloatTensor(np.concatenate((dep, dep, dep), axis=1)).to(self.device)
        return self.Actor_eval(rgb, dep, state).cpu().data.numpy()

    def learn(self):
        self.f += 1
        self.f %= 5
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor((b_memory[:, :3]).reshape(-1, 3)).to(self.device)
        b_a = torch.FloatTensor((b_memory[:, 3:6]).reshape(-1, 3)).to(self.device)
        b_r = torch.FloatTensor((b_memory[:, 6:7]).reshape(-1, 1)).to(self.device)
        b_s_ = torch.FloatTensor((b_memory[:, 7:10]).reshape(-1, 3)).to(self.device)

        # Compute the target Q value
        target_Q = self.Critic_target(self.rgb, self.dep, b_s_, self.Actor_target(self.rgb, self.dep, b_s_))
        target_Q = b_r + (GAMMA * target_Q).detach()

        # Get current Q estimate
        current_Q = self.Critic_eval(self.rgb, self.dep, b_s, b_a)

        # Compute critic loss
        critic_loss = self.loss_td(current_Q, target_Q)

        # Optimize the critic
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()

        # Compute actor loss
        ac = self.Critic_eval(self.rgb, self.dep, b_s, self.Actor_eval(self.rgb, self.dep, b_s)).mean()
        actor_loss = 100/(ac * ac)
        if self.f == 0:
            print(ac)
        # Optimize the actor
        self.atrain.zero_grad()
        actor_loss.backward()
        self.atrain.step()


        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)

        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

    def store_transition(self, s, a, r, s_):

        s = np.array(s).reshape(-1, 3)
        a = np.array(a).reshape(-1, 3)
        r = np.array(r).reshape(-1, 1)
        s_ = np.array(s_).reshape(-1, 3)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )
