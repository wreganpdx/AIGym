import gym
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from DataInit import DataInit
from NetworkSig import NetworkSig
from HiddenNetwork import HiddenNetwork
from SampleSpace import SampleSpace
from gym_recording.wrappers import TraceRecordingWrapper
from BestScores import BestScores



from gym import envs
global env
print(envs.registry.all())

from gym import spaces

exercise, numMinutes, render, printStuff, obsDefault = DataInit.getData()

best = BestScores(exercise)



s_folder = exercise
t = time.time()
s_file = str(t) + "chart"
s_file_weights = str(t) + "weights"
print(s_file)
print(s_file_weights)
if os.path.exists(s_folder) != True:
    os.mkdir(s_folder)

if os.path.exists(s_folder + "/weights") != True:
    os.mkdir(s_folder + "/weights")


env = gym.make(exercise)
#env = TraceRecordingWrapper(env, s_folder)
env.reset()
obs_space, action_space = SampleSpace.sampleSpace(env)

action = env.action_space.sample()
obs, r, d, i = env.step(action)

print(action_space)
#first paramater is number of hidden layer neurons. 
#hypothesis is that 2 is better than 1 because
#2nd layer can be dedicated to delta OBs
#which is last obs - current obs
#while first layer will only use current obs  
#think of it as general state vs what's new
#in theory, delta will capture things like momentum 
#and delta will eliminate things that are static.
#while current obs will give more stronger signals for things
#that are always the same
net = NetworkSig(action_space, action_space, obs_space,obs, 1, "dObs", True)

b = '%d' % int(best.getBest())
#b = "310"

NetworkSig.loadWeightsFromDisk(exercise+ "/weights/" + b)

env.reset()
numMinutes = int(numMinutes)

global rew
global episodes
global steps
global bestReward
global curSteps
global lastReward
global scores
global bestSurvival
bestSurvival = 0
rew = 0
steps = 1
episodes = 0
lastReward = 0
best_survival = 0
steps = 0
curSteps = 0
bestReward = best.getBest()
eps = []
scores = []
duration = numMinutes
duration = duration * 60 
render_t = time.time()
start = time.time()

lives = env.ale.lives()

stale = 0
lastScore = 5000

def saveBestStuff():
    global rew
    global episodes
    global steps
    global bestReward
    global bestSurvival
    
    bestReward = rew
    best.setBest(bestReward)
    bestSurvival = curSteps
    net.saveBest()
    best.save()
    b = '%d'%int(bestReward)
    NetworkSig.saveWeightsToDisk(exercise+ "/weights/" + b)
    print("save best: ", bestReward)

def resetLevel():
    global lastReward
    global curSteps
    global episodes
    global scores
    global rew
    global env
    lastReward = rew
    env.close()
    env = gym.make(exercise)
    env.reset()
    env.seed(0)
    env.action_space.np_random.seed(0)
 
    episodes += 1
    eps.append(episodes)
    scores.append(rew)
    if printStuff:
        print("reward: ", rew)
	print("steps: %d seconds %2.4f"% (steps, T))
	print("curSteps: %d"% curSteps)
    rew = 0
    curSteps = 0

env.close()
env = gym.make(exercise)
env.reset()
env.seed(0)
env.action_space.np_random.seed(0)
action = env.action_space.sample()
while True:
    steps += 1
    curSteps += 1
    t = time.time()
    T = t - start
    if T > duration:
	print("Done - T Seconds %d", T)
	break
    if render:
	env.render()

    observation, reward, done, info = env.step(action)
    stale += 1
    if done != True and env.ale.lives() == lives and stale < lastScore * 2:
	if reward > 0:
	    lastScore = stale
	    stale = 0
	    reward = 1
        rew += reward
    else:
        if rew > bestReward or (rew == bestReward and curSteps > bestSurvival):
	    print("rew : %d, bestReward: %d, curSteps:%d, best_survival%d"%(rew,bestReward,curSteps,bestSurvival))
            saveBestStuff()
        else:
            net.applyAnnealing(T, duration, rew > lastReward, scores)
	resetLevel()
    rand = env.action_space.sample()
    action = net.calculateObs(observation, rand)

#np.savetxt(s_folder + "/" + s_file, np.array([[1, 2], [3, 4]]), fmt="%s")
print('bestscore :%d', bestReward)
print('episodes :%d', episodes)

NetworkSig.restoreBest()
b = '%d'%int(bestReward)

_label0 = exercise #nice formating
plt.xlabel("Episodes")
plt.ylabel("Reward %")
plt.plot(eps, scores, color='c', label=_label0)#plot graph (note won't work if also doing confusion matrix)
plt.legend(loc="best")
env.close()
plt.savefig(s_folder + "/" + s_file, format='jpg')
plt.figure()
plt.show()
