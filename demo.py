import gym
import numpy as np
import os
import time

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
IF_RENDER = True
ENV_NAME = 'BipedalWalkerHardcore-v2'
EPISODE = 100
MAX_STEP = 20
ACTION_DIM = 0
STATE_DIM = 0
MEMORY_SIZE = 3000
LEARNING = False

if __name__ == "__main__":
    print('Current directory: ' + str(DIRECTORY))
    env = gym.make(ENV_NAME)
    ACTION_DIM = env.action_space.shape[0]
    STATE_DIM = env.observation_space.shape[0]
    for episode in range(EPISODE):
        start = time.time()
        state = env.reset()
        ep_reward = 0.
        ep_info = 0.
        for step in range(MAX_STEP):
            if IF_RENDER:
                env.render()
            action = np.zeros(shape=(ACTION_DIM, ))
            # ----here is the choose action -----

            # -----------------------------------
            next_state, reward, done, info = env.step(action)

            # -------store the transition -------

            # -----------------------------------
            if LEARNING:
                # ---------- learn --------------
                pass
            if done or step+1 == MAX_STEP:
                end = time.time()
                print('episode{0}: time={1:.2f}'.format(episode, end-start))
                break

