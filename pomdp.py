import torch
import torch.nn as nn
import torch.optim as optim
import ou_noise
import numpy as np
import time
import actor
import critic
import gym

ENV_NAME = 'BipedalWalkerHardcore-v2'
REPLAY_BUFFER_SIZE = int(1e5)
BATCH_SIZE = 30
GAMMA = 0.99
LR_C = 0.001
LR_A = 0.001


class DDPG_LSTM:
    def __init__(self, env):
        """

        :rtype: object
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = BATCH_SIZE
        self.name = 'POMDP'
        self.environment = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.replay_buffer = np.zeros((REPLAY_BUFFER_SIZE, self.state_dim * 2 + 1 + 2 + self.action_dim))
        self.buffer_counter = 0  # 记忆库计数
        self.noise = ou_noise.OUNoise(action_dimension=self.action_dim, theta=0.023, sigma=0.02)
        self.Actor_eval = actor.ActorNet(self.state_dim, self.action_dim)
        self.Actor_target = actor.ActorNet(self.state_dim, self.action_dim)
        self.Critic_eval = critic.CriticNet(self.state_dim, self.action_dim)
        self.Critic_target = critic.CriticNet(self.state_dim, self.action_dim)
        self.ctrain = optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, state, actor_init_hidden_cm):
        state_agent = torch.from_numpy(state[:14].reshape(1, 14, 1).astype(np.float32))
        state_rider = torch.from_numpy(state[14:].reshape(1, 10, 1).astype(np.float32))

        action_value, actor_init_hidden_cm = self.Actor_eval.forward(state_agent, state_rider, actor_init_hidden_cm)
        action_value = action_value.cpu().data.numpy()

        return action_value+self.noise.noise(), actor_init_hidden_cm

    def store_transition(self, state, reward, action, next_state, step, episode):
        s = np.array(state).reshape(-1, self.state_dim)
        a = np.array(action).reshape(-1, self.action_dim)
        r = np.array(reward).reshape(-1, 1)
        s_ = np.array(next_state).reshape(-1, self.state_dim)
        st = np.array(step).reshape(-1, 1)
        e = np.array(episode).reshape(-1, 1)

        transition = np.hstack((s, a, r, s_, st, e))
        index = self.buffer_counter % REPLAY_BUFFER_SIZE
        self.replay_buffer[index, :] = transition
        self.buffer_counter += 1

    def state_initializer(self, shape, mode='g'):
        if mode == 'z':  # Zero
            initial = np.zeros(shape=shape)
        elif mode == 'g':  # Gaussian
            initial = np.random.normal(loc=0.,scale=1./float(shape[1]),size=shape)
        else:  # May do some adaptive initializer can be built in later
            raise NotImplementedError
        return initial.astype(np.float32)

    def learning(self):
        pass

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = DDPG_LSTM(env)
    state = np.zeros(shape=(24,))
    state = env.reset()
    print(state.shape)
    hidden = (np.zeros(1, 1, 256), np.zeros(1, 1, 256))
    hidden = None
    action, hidden = agent.choose_action(state, hidden)
    print(action.shape)
    print(hidden[0].shape)
    print(hidden[0].max())
    print(hidden[1].shape)
    print(hidden[1].max())
