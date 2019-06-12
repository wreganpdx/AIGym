import gym
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from DataInit import DataInit
from NetMachineDvt import NetMachineDvt
from HiddenNetwork import HiddenNetwork
from SampleSpace import SampleSpace
from gym_recording.wrappers import TraceRecordingWrapper
from BestScores import BestScores



from gym import envs
print(envs.registry.all())

from gym import spaces
global env
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
net = NetMachineDvt(action_space, action_space, obs_space,obs, 1, "dObs", True)

unwrapped = env.unwrapped
if hasattr(unwrapped, 'get_action_meanings'):
    net.setClassLabels(env.unwrapped.get_action_meanings())
else: 
    net.setClassLabels(np.arange(action_space).astype('S10'))

b = '%.4f' % best.getBest()
#b = "310"
net.loadWeightsFromDisk(exercise+ "/weights/" + b)
net.setPath(exercise+ "/weights/")
net.setScoreObj(best)
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
global oldSeed
oldSeed = time.time()
bestSurvival = 0
rew = 0
steps = 1
episodes = 0
lastReward = 0
best_survival = 0
steps = 0
curSteps = 0
eps = []
scores = []
duration = numMinutes
duration = duration * 60 
render_t = time.time()
start = time.time()

if hasattr(env, 'ale'):
    lives = env.ale.lives()
else:
    lives = 0
def resetLevel(finishedSim, newSeed):
    global lastReward
    global curSteps
    global episodes
    global scores
    global rew
    global env
    global oldSeed
    lastReward = rew
    if newSeed:
	newSeed = time.time()
	oldSeed = newSeed
    else:
	newSeed = oldSeed
    
    if finishedSim:
        env.close()
        env = gym.make(exercise)
        env.reset()
        env.seed(newSeed)
        env.action_space.np_random.seed(newSeed)
    else:
	env.reset()
 
    episodes += 1
    eps.append(episodes)
    scores.append(rew)
    if printStuff:
        print("reward: ", rew)
	print("steps: %d seconds %2.4f"% (steps, T))
	print("curSteps: %d"% curSteps)
    rew = 0
    curSteps = 0

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
    if hasattr(env, 'ale'):
        curLives = env.ale.lives()
    else:
        curLives = 0
    observation, reward, done, info = env.step(action)
    if done != True and curLives == lives:
        rew += reward
    else:
        finishedSim, newSeed = net.applyAnnealing(T, duration, rew, scores)
	resetLevel(finishedSim, newSeed)
    rand = env.action_space.sample()
    action = net.calculateObs(observation, rand)

#np.savetxt(s_folder + "/" + s_file, np.array([[1, 2], [3, 4]]), fmt="%s")
print('bestscore :%d', np.max(scores))
print('episodes :%d', episodes)


plt.xlabel("Episodes")
plt.ylabel("Reward %")
plt.plot(eps, scores, color='c', label=exercise)
plt.legend(loc="best")
env.close()
plt.savefig(s_folder + "/" + s_file, format='jpg')
plt.figure()
plt.show()
