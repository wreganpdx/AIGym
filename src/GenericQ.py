"""
William C Regan
Student ID: 954681718
Artificial Intelligence
Portland State University
Final Project
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from DataInitQ import DataInitQ
from QNetwork import QNetwork
from SampleSpace import SampleSpace
import time

from gym import envs
#print(envs.registry.all())

from gym import spaces
exercise, numMinutes, do_render, stacks, diff, clusters, test, discrete, elitism, epsilon  = DataInitQ.getData()
print("Exercise: %s"% exercise)
print("Minutes: %s"% numMinutes)
print("Render: %r"% do_render)
print("Observation Stack Size: %d"% stacks)
print("Use Observation Difference: %r"%diff)
print("Number of clusters: %d"% clusters)
env = gym.make(exercise)
env.reset()
obs_space, action_space = SampleSpace.sampleSpace(env)

print("Action space: "),
print(action_space)
print("Observation space: "),
print(obs_space)
net = QNetwork(clusters, action_space, obs_space, stacks, diff, test, discrete, elitism, epsilon)
numMinutes = int(numMinutes)
duration = numMinutes
duration = duration * 60
start = time.time()

action = env.action_space.sample()
rew = 0
steps = 1
episodes = 0
lastReward = 0
bestReward = -50000000
eps = []
scores = []
reset = False
while True:
    t = time.time()
    T = t - start
    if T > duration:
        print("Done - T Seconds %d", T)
        break
    if do_render:
        env.render()
    observation, reward, done, info = env.step(action)
    if done != True:
        rew += reward
    else:
        if rew > bestReward:
            bestReward = rew
        else:
           net.applyAnnealing(T, duration, reward, rew)

        lastReward = rew
        env.reset()
        episodes += 1
        eps.append(episodes)
        scores.append(rew)
        #print("Minutes: ", T/60),
       # print("reward: ", rew)
        rew = 0
        reset = True

    if reset:
        action = 0
        reset = False
    else:
        action = net.calculateObs(observation, reward, T, duration)
        #action = env.action_space.sample()


s = []
e = []
length = len(eps)/100
for i in range(100):
    start = i * length
    end = (i+1) * length
    if end >= len(eps):
        end = len(eps)-1
    e.append(end)
    s.append(np.average(scores[i*length:(i+1)*length]))



print('bestscore :%d', bestReward)
print('episodes :%d', episodes)
_label0 = exercise #nice formating
plt.xlabel("Episodes")
plt.ylabel("Reward %")
plt.plot(e, s, color='c', label=_label0)#plot graph (note won't work if also doing confusion matrix)
plt.legend(loc="best")
env.close()
plt.figure()
plt.show()