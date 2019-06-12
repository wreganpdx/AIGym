import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")
env.reset()
rew = 0
train = np.random.rand(4)
train -= 1.0
train *= 2.0
print(train)
action = np.dot(train, train.transpose())

bestscore = 0
episodes = 0
eps = []

scores = []
stepNum = 50000
delta = np.random.rand(4)
delta -= 1.0
delta *= .01

bestTrain = train[:]

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


for T in range(stepNum):
    env.render()
    dec = sigmoid(action)
    if dec > .5:
        action = 1
    else:
        action = 0
    observation, reward, done, info = env.step(action)
    rew += reward
   # print(observation)
    if (done):
        episodes += 1
        eps.append(episodes)
        env.reset()
        scores.append(rew)



        avg = 0

        r = episodes /2
        if r > 5:
            r = 5
        for i in range(r):
            avg += scores[-i]
        notZero = r
        if notZero == 0:
            notZero = 1
        avg = avg/notZero

       # if episodes > 1:
         #   avg = scores[-1]
        #print('environment ended after :%d', steps)

        scale = np.abs((rew - avg))
        if scale < .01:
            scale = .01

        if rew >= bestscore:
            print('Saving Best Weights old:', bestscore, bestTrain)
            bestTrain = train + delta
            bestscore = rew
            print('Saving Best Weights new:', rew, bestTrain)

        if rew > avg:
            print('Reward', rew, avg, episodes)
            train = train + delta
        rand = np.random.rand();
        if rand > float(T)/float(stepNum):
            if rew < avg:
                print('Punish', rew, avg, episodes)
                train = train - delta
        else:
            rand = np.random.rand();
            if rand > float(T) / float(stepNum):
                print('Random Kick Weights', rew, avg, episodes)
                train = np.random.rand(4)
            else:
                train = bestTrain[:]
                print('Restoring Best Weights', rew, avg, episodes)
                print(train)
        rew = 0
        delta = np.random.rand(4)
        delta -= 1.0
        delta *= .1
    else:
        obs = observation[:]
        action = np.dot( (train + delta), obs.transpose())

print('bestscore :%d', bestscore)
print('episodes :%d', episodes)
_label0 = "Pole Physices" #nice formating
plt.xlabel("Episodes")
plt.ylabel("Reward %")
plt.plot(eps, scores, color='c', label=_label0)#plot graph (note won't work if also doing confusion matrix)
plt.legend(loc="best")
env.close()
plt.figure()
plt.show()