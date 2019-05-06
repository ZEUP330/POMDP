import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

# Hyper-Parameters
LAYER1_SIZE = 192
LEARNING_RATE = 0.001
INPUT_SIZE = 100
NUM_RNN_LAYER = 4
TAU = 0.
OUT_PUT_SIZE = 256


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.out_put_size = OUT_PUT_SIZE
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=1, stride=LAYER1_SIZE,)
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2 = nn.Conv2d(64, 128, 1, LAYER1_SIZE, )
        self.conv2.weight.data.normal_(0, 0.1)
        self.conv3 = nn.Conv2d(10, 64, 1, LAYER1_SIZE, )
        self.conv3.weight.data.normal_(0, 0.1)
        self.conv4 = nn.Conv2d(64, 128, 1, LAYER1_SIZE, )
        self.conv4.weight.data.normal_(0, 0.1)
        self.rnn = nn.LSTM(input_size=256,
                           hidden_size=OUT_PUT_SIZE,
                           num_layers=NUM_RNN_LAYER,
                           batch_first=True)
        self.out = nn.Linear(OUT_PUT_SIZE, action_dim)
        self.last_epi = -1

    def forward(self, state_agent, state_rider, hidden_cm):
        x = self.conv1(state_agent)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        y = self.conv3(state_rider)
        y = F.relu(y)
        y = self.conv4(y)
        y = F.relu(y)
        x = torch.cat((x.float(), y.float()), dim=1)
        x = x.reshape(-1, 1, 256)
        x, hidden_cm = self.rnn(x, hidden_cm)
        actions_value = self.out(x)
        return actions_value, hidden_cm
