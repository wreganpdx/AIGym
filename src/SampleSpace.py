import gym
import numpy as np

class SampleSpace(object):

    def __init__(self):
        return None

    @staticmethod
    def sampleSpace(env):

        env.reset()
        actions = []
        for i in range(5000):
            actions.append(env.action_space.sample())

        actions = np.unique(actions)

        a = env.action_space.sample()

        observation, reward, done, info = env.step(actions[0])

        return (len(observation), len(actions))

