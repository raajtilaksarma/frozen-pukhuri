import sys
import gym
import numpy as np
import random
import time

def get_position(env):
    row, col = env.s // env.ncol, env.s % env.ncol
    return row, col


def attempting(attempts = 1):
    env = gym.make("FrozenLake-v0")
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    # print(q_table)

    num_episodes = 10000
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.005
    rewards_all_episodes = []
    # read / learn q-table
    q_table = np.load('q_table.npy')
    # positions = []
    for episode in range(attempts):
        state = env.reset()
        done = False
        print("Attempt Number : ",episode+1)
        time.sleep(0.5)

        for step in range(max_steps_per_episode):
    #         clear_output(wait=True)
            row, col = get_position(env)
            yield (row,col)
            # positions.append((row, col))
    #         get_position(env)
            time.sleep(0.1)
            action = np.argmax(q_table[state, :])
            new_state, reward, done, info = env.step(action)

            if done:
    #             clear_output(wait=True)
    #             get_position(env)
                if reward==1:
                    # print("Goal reached. . .")
                    yield 'Goal reached. . .'
                else:
                    # print("Mission failed. . .")
                    yield 'Mission failed. . .'
                break
            state = new_state
    env.close()
    # return positions

def train():
    env = gym.make("FrozenLake-v0")
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    # print(q_table)

    num_episodes = 10000
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001
    rewards_all_episodes = []

    for episode in range(num_episodes):
#     print('Episode no:'+str(episode+1))
        state = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps_per_episode):
            # trading off between exploration and exploitation
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate and ~np.all(q_table[state,:]==0):
                # exploit
                action = np.argmax(q_table[state, :])
            else:
                # explore
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            # update Q-table
            q_table[state, action] = q_table[state, action]*(1-learning_rate) + \
                                learning_rate * (reward + discount_rate*np.max(q_table[new_state, :]))
            state = new_state
            rewards_current_episode += reward
            if done==True:
                break
        exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)

    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
    count = 1000
    print('Avg reward per thousand episodes. . .')
    for r in rewards_per_thousand_episodes:
        print(str(count) + " : " + str(sum(r)/1000) )
        count+=1000
    print('Final Q-table. . .')
    print(q_table)
