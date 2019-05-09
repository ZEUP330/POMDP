from typing import Tuple, Union

import gym
import numpy as np
import torch
import os
import time
import actor

from numpy.core._multiarray_umath import ndarray

import pomdp

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
IF_RENDER = True
ENV_NAME = 'BipedalWalkerHardcore-v2'
EPISODE = 1000
MAX_STEP = 200  # self.environment.spec.max_episode_steps = 2000
ACTION_DIM = 4
STATE_DIM = 24
MEMORY_SIZE = 3000
LEARNING = False

if __name__ == "__main__":
    print('Current directory: ' + str(DIRECTORY))
    env = gym.make(ENV_NAME)
    agent = pomdp.DDPG_LSTM(env)
    ACTION_DIM = env.action_space.shape[0]
    STATE_DIM = env.observation_space.shape[0]
    best_reward = 365.
    for episode in range(EPISODE):
        start = time.time()
        state = env.reset()
        ep_reward = 0.
        ep_info = 0.
        init_actor_hidden1_c = agent.state_initializer(shape=(actor.NUM_RNN_LAYER, 1, agent.Actor_eval.out_put_size), mode='g')
        # output(or cell) hidden state
        init_actor_hidden1_m = agent.state_initializer(shape=(actor.NUM_RNN_LAYER, 1, agent.Actor_eval.out_put_size), mode='g')
        # memory hidden state
        actor_init_hidden_cm = (torch.from_numpy(init_actor_hidden1_c), torch.from_numpy(init_actor_hidden1_m))

        for step in range(MAX_STEP):
            if IF_RENDER and pomdp.REPLAY_BUFFER_SIZE < agent.buffer_counter:
                env.render()
            # ----here is the choose action -----
            action, actor_last_hidden_cm = agent.choose_action(state, actor_init_hidden_cm)
            action = action.reshape(-1, action.size)
            next_state, reward, done, info = env.step(action[0])

            ep_reward += reward
            if step + 1 == MAX_STEP:
                done = True

            # ### for test
            # state = np.array([step for x in range(24)])
            # action = np.array([step for x in range(4)])
            # next_state = np.array([step+1 for x in range(24)])
            # ### for test

            # -------store the transition -------
            agent.store_transition(state, reward, action, next_state, done, step, episode)

            # ### for test
            # if episode == 5 and step == 4:
            #     agent.learning(step)
            # ### for test

            if pomdp.REPLAY_BUFFER_SIZE < agent.buffer_counter:
                agent.learning(step)

            state = next_state
            actor_init_hidden_cm = actor_last_hidden_cm

            if done:
                end = time.time()
                print('episode{0}:reward:{1}, time={2:.2f}'.format(episode, ep_reward, end-start))
                if best_reward < ep_reward:
                    best_reward = ep_reward
                break

