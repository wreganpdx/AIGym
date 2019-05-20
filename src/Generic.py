import gym
import numpy as np
import matplotlib.pyplot as plt
from DataInit import DataInit
from Network import Network
from HiddenNetwork import HiddenNetwork
from SampleSpace import SampleSpace

from gym import envs
print(envs.registry.all())

from gym import spaces

exercise, numIterations = DataInit.getData()
env = gym.make(exercise)
env.reset()
obs_space, action_space = SampleSpace.sampleSpace(env)


print(action_space)
net = Network(action_space * 1, action_space, obs_space)

numIterations = int(numIterations)
action = env.action_space.sample()
rew = 0
steps = 1
episodes = 0
lastReward = 0
bestReward = -50000000
eps = []
scores = []
for T in range(numIterations):
    env.render()
    observation, reward, done, info = env.step(action)
    if done != True:
        rew += reward
       # print(reward),
    else:
        if rew > bestReward:
            bestReward = rew
            net.saveBest()
            print("save best: ", bestReward)
        else:
            net.applyAnnealing(T, numIterations, rew > lastReward)

        lastReward = rew
        env.reset()
        episodes += 1
        eps.append(episodes)
        scores.append(rew)
        print("reward: ", rew)
        print("steps: ", T)
        rew = 0


    action = net.calculateObs(observation)


print('bestscore :%d', bestReward)
print('episodes :%d', episodes)
_label0 = exercise #nice formating
plt.xlabel("Episodes")
plt.ylabel("Reward %")
plt.plot(eps, scores, color='c', label=_label0)#plot graph (note won't work if also doing confusion matrix)
plt.legend(loc="best")
env.close()
plt.figure()
plt.show()