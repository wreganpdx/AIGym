"""
William C Regan
Student ID: 954681718
Machine Learning
Portland State University
Final Project
"""
import numpy as np
import random
from Util import Util
import time
class Neuron(object):

    def __init__(self, clas, seed, weights):
        self.weights = weights
        self.clas = clas
        self.seed = seed
        self.seedActual = int(time.time())%((clas**4)+2)
        print("Creating neuron class: %d seed: %d" %(clas, self.seedActual))
        np.random.seed(self.seedActual)  # place a random seed (actually it's just the index, but it works since each perceptron has its own index)
        self.seed = self.clas
        self.delta =  np.zeros(weights)
	self.w = np.zeros(weights)
	for i in range(weights):
	   self.w[i] = 1.0/(float(weights)*2)
	self.w[clas] = .5
        self.fitness = 0
        self.wCopy = self.w[:]
	


    def getClass(self):
        return self.clas

    def copyWeights(self):
        np.copyto(self.wCopy, self.w)


    def restoreWeights(self):
        np.copyto(self.w,self.wCopy)


    def backPropagate(self,reward):
        if reward == 0:
            self.backpropPunish()
        else:
            self.backpropReward()


    def backpropPunish(self):
        self.w = self.w + self.delta * -1

    def backpropReward(self):
        self.w = self.w + self.delta

    def resetDelta(self):
        self.delta = np.random.rand(self.weights)
        self.delta -= 1.0
        self.delta *= .0001

    def anealDelta(self, a, rew):
        self.delta = np.random.rand(self.weights)
        self.delta *= .0001
	self.delta *= rew
	self.delta *= (1-a)
    def anealSigmoid(self, a, rew, sig, result):
	if rew == 1:
	    if sig > .9:
		return
	    else:
                diff = .9 - rew
	else:
	    if sig < .1:
		return
            else:
		diff = rew - .1
	prime = Util.sigmoid_prime(diff)	
	self.delta = prime * result * rew * (1-a)
 
    def zeroDelta(self):
        self.delta = self.delta * 0

    def punishDelta(self, delta, avg):
	if avg < .1:
	    return
	diff = avg - .1
	gradient = Util.sigmoid_prime(diff)
	self.delta = self.delta + (delta * gradient * -.1)

    def punishDeltaRC(self, delta, avg, LearningRate):
	if avg < .1:
	    return False
	diff = avg - .1
	gradient = Util.sigmoid_prime(diff)
	self.delta = self.delta + (delta * gradient * LearningRate * -1)
	return True

    def rewardDeltaRC(self, delta, avg, LearningRate):
	if avg > .9:
	   return False
	diff = .9 - avg
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * LearningRate * 1)
	return True

    def rewardDelta(self, delta, avg):
	if avg > .9:
	   return
	diff = .9 - avg
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * .1)


    def resetFitness(self):
        self.fitness = 0

    def compute(self, obs):
        self.personalPrediction = Util.sigmoid(np.dot(self.w + self.delta, obs.transpose())  )
        self.prediction = self.personalPrediction
        return self.prediction


