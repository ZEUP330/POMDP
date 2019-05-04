import gym
import numpy as np
import os
import time
import pomdp

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
IF_RENDER = True
ENV_NAME = 'BipedalWalkerHardcore-v2'
EPISODE = 10
MAX_STEP = 20
ACTION_DIM = 0
STATE_DIM = 0
MEMORY_SIZE = 3000
LEARNING = False

if __name__ == "__main__":
    print('Current directory: ' + str(DIRECTORY))
    env = gym.make(ENV_NAME)
    agent = pomdp.DDPG_LSTM(env)
    ACTION_DIM = env.action_space.shape[0]
    STATE_DIM = env.observation_space.shape[0]
    for episode in range(EPISODE):
        start = time.time()
        state = env.reset()
        ep_reward = 0.
        ep_info = 0.
        init_actor_hidden1_c = agent.state_initialiser(shape=(1, agent.Actor_eval.rnn_size), mode='g')
        # output(or cell) hidden state
        init_actor_hidden1_m = agent.state_initialiser(shape=(1, agent.Actor_eval.rnn_size), mode='g')
        # memory hidden state
        actor_init_hidden_cm = (init_actor_hidden1_c, init_actor_hidden1_m)
        for step in range(MAX_STEP):
            if IF_RENDER:
                env.render()
            # ----here is the choose action -----
            action, actor_last_hidden_cm = agent.choose_action(state, actor_init_hidden_cm)

            next_state, reward, done, info = env.step(action)

            # -------store the transition -------
            agent.store_transition(state, reward, action, next_state)

            if pomdp.REPLAY_BUFFER_SIZE == agent.buffer_counter:
                agent.learning()

            state = next_state
            actor_init_hidden_cm = actor_last_hidden_cm

            if done or step+1 == MAX_STEP:
                end = time.time()
                print('episode{0}: time={1:.2f}'.format(episode, end-start))
                break

