import torch
import torch.nn as nn
import torch.optim as optim
import ou_noise
import numpy as np
import os
import time
import actor
import critic
import gym
import replay_buffer

GAMMA = 0.99
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
ENV_NAME = 'BipedalWalkerHardcore-v2'
REPLAY_BUFFER_SIZE = int(1e5)
BATCH_SIZE = 30
OPT_LENGTH = 90
LR_C = 0.001
LR_A = 0.001
TRACE_LENGTH = 9
NUM_RNN_LAYER = 2


class DDPG_LSTM(object):
    def __init__(self, env):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = BATCH_SIZE
        self.name = 'POMDP'
        self.environment = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_len_trajectory = self.environment.spec.max_episode_steps + 1
        self.noise = ou_noise.OUNoise(action_dimension=self.action_dim, theta=0.023, sigma=0.02)
        self.Actor_eval = actor.ActorNet(self.state_dim, self.action_dim)
        self.Actor_target = actor.ActorNet(self.state_dim, self.action_dim)
        self.Critic_eval = critic.CriticNet(self.state_dim, self.action_dim)
        self.Critic_target = critic.CriticNet(self.state_dim, self.action_dim)
        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_SIZE,
                                                        DIRECTORY, self.max_len_trajectory, self.Actor_eval.last_epi)
        # self.replay_buffer = np.zeros((REPLAY_BUFFER_SIZE, self.state_dim * 2 + 1 + 2 + self.action_dim))
        self.buffer_counter = 0  # 记忆库计数
        self.ctrain = optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.trace_length = TRACE_LENGTH

        # (num_layers * num_directions, mini_batch, hidden_size[out_put size])
        self.hidden_a = torch.from_numpy(self.state_initializer(shape=(actor.NUM_RNN_LAYER, BATCH_SIZE,
                                                                       self.Actor_eval.out_put_size), mode='z'))
        self.hidden_c = torch.from_numpy(self.state_initializer(shape=(actor.NUM_RNN_LAYER, BATCH_SIZE,
                                                                       self.Critic_eval.out_put_size), mode='z'))
        # (num_layers * num_directions, mini_batch, hidden_size[out_put size])

        self.target_actor_init_h_batch = self.actor_init_h_batch = (self.hidden_a, self.hidden_a)
        self.target_critic_init_h_batch = self.critic_init_h_batch = (self.hidden_c, self.hidden_c)
        self.discounting_mat_dict = {}

    def choose_action(self, state, actor_init_hidden_cm):

        state_agent = torch.from_numpy(np.expand_dims(np.array(state[:14]).reshape(1, 14, 1).astype(np.float32),axis=2))
        state_rider = torch.from_numpy(np.expand_dims(np.array(state[14:]).reshape(1, 10, 1).astype(np.float32), axis=2))
        action_value, actor_init_hidden_cm = self.Actor_eval.forward(state_agent, state_rider, actor_init_hidden_cm)
        action_value = action_value.cpu().data.numpy()
        return action_value+self.noise.noise(), actor_init_hidden_cm

    def store_transition(self, state, reward, action, next_state, done, step, episode):
        done = float(done)
        self.replay_buffer.add(state, action, reward, next_state, done, episode, step)
        if done:
            self.noise.reset()
        #
        # s = np.array(state).reshape(-1, self.state_dim)
        # a = np.array(action).reshape(-1, self.action_dim)
        # r = np.array(reward).reshape(-1, 1)
        # s_ = np.array(next_state).reshape(-1, self.state_dim)
        # st = np.array(step).reshape(-1, 1)
        # e = np.array(episode).reshape(-1, 1)
        #
        # transition = np.hstack((s, a, r, s_, st, e))
        # index = self.buffer_counter % REPLAY_BUFFER_SIZE
        # self.replay_buffer[index, :] = transition
        self.buffer_counter += 1

    def state_initializer(self, shape, mode='g'):
        if mode == 'z':  # Zero
            initial = np.zeros(shape=shape)
        elif mode == 'g':  # Gaussian
            initial = np.random.normal(loc=0., scale=1./float(shape[1]), size=shape)
        else:  # May do some adaptive initializer can be built in later
            raise NotImplementedError
        return initial.astype(np.float32)

    def np_to_list(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = data[i][j].tolist()
        return data

    def learning(self, step):
        # *************************************get mini-batch of transition*********************************************
        minibatch, trace_length = self.replay_buffer.get_batch(batch_size=BATCH_SIZE,
                                                               max_trace_length=self.trace_length,
                                                               time_step=step)
        try:
            state_trace_batch = np.array(minibatch[:, :, 2].tolist())
            state_trace_batch = np.stack(state_trace_batch.ravel()).reshape(self.batch_size,
                                                                             trace_length, self.state_dim)
            action_trace_batch = np.stack(minibatch[:, :, 3].ravel()).reshape(self.batch_size,
                                                                              trace_length, self.action_dim)
            next_state_batch = np.stack(minibatch[:, -1, 6].ravel()).reshape(self.batch_size,
                                                                             1, self.state_dim)
            next_state_trace_batch = np.concatenate([state_trace_batch, next_state_batch], axis=1)
            reward_trace_batch = np.stack(minibatch[:, :, 4].ravel()).reshape(self.batch_size, trace_length, 1)
            done_trace_batch = np.stack(minibatch[:, :, 7].ravel()).reshape(self.batch_size, trace_length, 1)
        except Exception as e:
            print(str(e))
            raise
        # ***************************Painfully initialise initial memories of LSTMs:************************************
        # not super-efficient, but no error guaranteed from tf's None-type zero-state problem
        self.actor_init_h_batch = (self.hidden_a, self.hidden_a)
        self.critic_init_h_batch = (self.hidden_c, self.hidden_c)
        if trace_length <= OPT_LENGTH:
            self.target_actor_init_h_batch = self.actor_init_h_batch
            self.target_critic_init_h_batch = self.critic_init_h_batch
            pass
        else:
            state_agent = state_trace_batch[:, :-OPT_LENGTH, :14].reshape(BATCH_SIZE, 14, -1).astype(np.float32)
            state_agent = np.expand_dims(state_agent, axis=3)
            state_agent = torch.from_numpy(state_agent)
            state_rider = state_trace_batch[:, :-OPT_LENGTH, 14:].reshape(BATCH_SIZE, 10, -1).astype(np.float32)
            state_rider = np.expand_dims(state_rider, axis=3)
            state_rider = torch.from_numpy(state_rider)
            not_use, self.actor_init_h_batch = self.Actor_eval.forward(
                        state_agent, state_rider, self.actor_init_h_batch)
            self.target_actor_init_h_batch = self.actor_init_h_batch

            action = action_trace_batch[:, :-OPT_LENGTH, :].reshape(BATCH_SIZE, 4, -1).astype(np.float32)
            action = np.expand_dims(action, axis=3)
            action = torch.from_numpy(action)
            not_use, self.critic_init_h_batch = self.Critic_eval.forward(
                        state_agent, state_rider, action, self.critic_init_h_batch)
            self.target_critic_init_h_batch = self.critic_init_h_batch

            state_trace_batch = state_trace_batch[:, -OPT_LENGTH:, :]
            next_state_trace_batch = next_state_trace_batch[:, -(OPT_LENGTH+1):, :]
            action_trace_batch = action_trace_batch[:, -OPT_LENGTH:, :]
            reward_trace_batch = reward_trace_batch[:, -OPT_LENGTH:, :]
            done_trace_batch = done_trace_batch[:, -OPT_LENGTH:, :]

        # **************************************Obtain target output****************************************************
        state_agent_t = next_state_trace_batch[:, :, :14].reshape(BATCH_SIZE, 14, -1).astype(np.float32)
        state_agent_t = np.expand_dims(state_agent_t, axis=3)
        state_rider_t = next_state_trace_batch[:, :, 14:].reshape(BATCH_SIZE, 10, -1).astype(np.float32)
        state_rider_t = np.expand_dims(state_rider_t, axis=3)
        state_agent_t = torch.from_numpy(state_agent_t)
        state_rider_t = torch.from_numpy(state_rider_t)
        next_action_batch, not_use = self.Actor_target.forward(
                    state_agent_t, state_rider_t, self.target_actor_init_h_batch)
        next_action_trace_batch = np.concatenate([action_trace_batch, next_action_batch.cpu().data.numpy()], axis=1)
        next_action_trace_batch = next_action_trace_batch.reshape(BATCH_SIZE, 4, -1)
        next_action_trace_batch = next_action_trace_batch.astype(np.float32)
        next_action_trace_batch = np.expand_dims(next_action_trace_batch, axis=3)
        next_action_trace_batch = torch.from_numpy(next_action_trace_batch)
        target_last_Q_batch, not_use = self.Critic_target.forward(
            state_agent_t, state_rider_t, next_action_trace_batch, self.target_critic_init_h_batch)

        # *********************************Control the length of time-step for gradient*********************************
        if trace_length <= OPT_LENGTH:
            update_length = np.minimum(trace_length, OPT_LENGTH // 1)  # //denom: 2(opt1) #1(opt0) #OPT_LENGTH(opt2)
        else:
            update_length = OPT_LENGTH // 1  # //denom: 2(opt1) #1(opt0) #OPT_LENGTH(opt2)
        target_lastQ_batch_masked = target_last_Q_batch[:, -1].cpu().data.numpy() * (1. - done_trace_batch[:, -1])
        rQ = np.concatenate([np.squeeze(reward_trace_batch[:, -update_length:], axis=-1),
                             target_lastQ_batch_masked], axis=1)
        try:
            discounting_mat = self.discounting_mat_dict[update_length]
        except KeyError:
            discounting_mat = np.zeros(shape=(update_length, update_length+1),dtype=np.float)
            for i in range(update_length):
                discounting_mat[i, :i] = 0.
                discounting_mat[i, i:] = GAMMA ** np.arange(0., -i+update_length+1)
            discounting_mat = np.transpose(discounting_mat)
            self.discounting_mat_dict[update_length] = discounting_mat
        try:
            mat = np.matmul(rQ, discounting_mat)
            y_trace_batch = np.expand_dims(mat, axis=-1)
        except Exception as e:
            print(str(e), '?')
            raise
        # ****************************Train Critic: get next_action, target_q, then optimise****************************
        y_trace_batch = torch.from_numpy(y_trace_batch.astype(np.float32))

        state_agent = state_trace_batch[:, :, :14].reshape(BATCH_SIZE, 14, -1).astype(np.float32)
        state_agent = np.expand_dims(state_agent, axis=3)
        state_agent = torch.from_numpy(state_agent)
        state_rider = state_trace_batch[:, :, 14:].reshape(BATCH_SIZE, 10, -1).astype(np.float32)
        state_rider = np.expand_dims(state_rider, axis=3)
        state_rider = torch.from_numpy(state_rider)
        action = action_trace_batch[:, :, :].reshape(BATCH_SIZE, 4, -1).astype(np.float32)
        action = np.expand_dims(action, axis=3)
        action = torch.from_numpy(action)
        Q_value, Critic_hidden_value = self.Critic_eval.forward(state_agent, state_rider, action, self.critic_init_h_batch)
        self.loss = self.loss_td(y_trace_batch, Q_value[:, -update_length:, :])  # + weight_decay
        # Optimize the critic
        self.ctrain.zero_grad()
        self.loss.backward()
        self.ctrain.step()
        i = 0
        # Train Actor:while updated critic,we declared the dQda. Hence sess,run(dQda*dadParam_actor),then optimise actor
        for i in range(update_length):  # todo in here the i all become 4
            actor_init_h_batch_stack = actor_init_h_batch_trace = (
            np.expand_dims(self.actor_init_h_batch[0], axis=2), np.expand_dims(self.actor_init_h_batch[1], axis=2))
            critic_init_h_batch_stack = critic_init_h_batch_trace = (
            np.expand_dims(self.critic_init_h_batch[0], axis=2), np.expand_dims(self.critic_init_h_batch[1], axis=2))
            if i == 0:
                actor_init_h_batch_stack = actor_init_h_batch_trace
                critic_init_h_batch_stack = critic_init_h_batch_trace
            else:
                actor_init_h_batch_stack = (
                    np.concatenate((actor_init_h_batch_stack[0], actor_init_h_batch_trace[0]), axis=2),
                    np.concatenate((actor_init_h_batch_stack[1], actor_init_h_batch_trace[1]), axis=2))
                critic_init_h_batch_stack = (
                    np.concatenate((critic_init_h_batch_stack[0], critic_init_h_batch_trace[0]), axis=2),
                    np.concatenate((critic_init_h_batch_stack[1], critic_init_h_batch_trace[1]), axis=2))
            agent_ex = np.expand_dims(np.expand_dims(state_trace_batch[:, i, :14], 2), 2)
            agent_ = torch.from_numpy(agent_ex.astype(np.float32))
            rider_ex = np.expand_dims(np.expand_dims(state_trace_batch[:, i, 14:], 2), 2)
            rider_ = torch.from_numpy(rider_ex.astype(np.float32))
            action_trace_batch_for_gradients, actor_init_h_batch = self.Actor_eval.forward(agent_, rider_,
                                                                                           self.actor_init_h_batch)
            action_ex = np.expand_dims(np.expand_dims(action_trace_batch[:, i], 2), 2)
            action_ = torch.from_numpy(action_ex.astype(np.float32))
            not_use, critic_init_h_batch = self.Critic_eval.forward(agent_, rider_, action_, self.critic_init_h_batch)
            if i == 0:
                action_trace_batch_for_gradients_stack = action_trace_batch_for_gradients.cpu().data.numpy()
            else:
                action_trace_batch_for_gradients_stack = np.concatenate(
                 (action_trace_batch_for_gradients_stack, action_trace_batch_for_gradients.cpu().data.numpy()), axis=1)

        state_trace_batch_stack = np.reshape(state_trace_batch, (self.batch_size*update_length, 1, self.state_dim))
        action_trace_batch_stack = np.reshape(action_trace_batch, (self.batch_size*update_length, 1, self.action_dim))
        action_trace_batch_for_gradients_stack = np.reshape(action_trace_batch_for_gradients_stack,
                                                            (self.batch_size*update_length, 1, self.action_dim))
        hidden_a1 = actor_init_h_batch_stack[0].reshape(actor.NUM_RNN_LAYER, self.batch_size*update_length, self.Actor_eval.out_put_size)
        hidden_a2 = actor_init_h_batch_stack[1].reshape(actor.NUM_RNN_LAYER, self.batch_size*update_length, self.Actor_eval.out_put_size)
        actor_init_h_batch_stack = (hidden_a1, hidden_a2)
        hidden_c1 = critic_init_h_batch_stack[0].reshape(actor.NUM_RNN_LAYER, self.batch_size*update_length, self.Critic_eval.out_put_size)
        hidden_c2 = critic_init_h_batch_stack[1].reshape(actor.NUM_RNN_LAYER, self.batch_size*update_length, self.Critic_eval.out_put_size)
        critic_init_h_batch_stack = (hidden_c1, hidden_c2)





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
