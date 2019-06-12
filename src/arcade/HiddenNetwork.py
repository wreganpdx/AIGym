"""
William C Regan
Machine Learning
Portland State University
Student ID: 954681718
Assignment 2
"""
import numpy as np
import random
from Util import Util
import time
class HiddenNetwork(object):

    def __init__(self, seed, clas, weights):
        self.weights = weights
        self.clas = clas
        self.seedActual = int(time.time())%(clas**5+2)
        print("Creating hidden neuron class: %d seed:"% self.seedActual)
        np.random.seed(self.seedActual)  

        self.rekick()
	self.resetDelta()
        self.wCopy = self.w[:]
        self.copyWeights()
	self.predictions = []
	self.maxSigs = 30

    def getClass(self):
        return self.clas

    def copyWeights(self):
        np.copyto(self.wCopy, self.w)


    def restoreWeights(self):
        np.copyto(self.w,self.wCopy)

        #print(self.w, self.personalBias),


    def backPropagate(self, reward):
        if reward == 0:
            self.backpropPunish()
        else:
            self.backpropReward()

    def resetDelta(self):
        self.delta = np.random.random_sample(self.weights)
        self.delta -= 1.0
        self.delta *= .0001

    def zeroDelta(self):
        self.delta = self.delta * 0

    def rekick(self):
        self.w = np.random.random_sample(self.weights)  # set weights
        self.w -= 1.0
        self.w *= .01
        self.resetDelta()


    def backpropPunish(self):
        self.w = self.w +  self.delta * -1
        self.resetDelta()

    def backpropReward(self):
        self.w = self.w + self.delta
        self.resetDelta()

    def punishDelta(self, delta):
        avg = np.average(self.predictions)
	if avg < .1:
	   return
	diff = avg - .1
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * -.1)

    def rewardDelta(self, delta):
	avg = np.average(self.predictions)
	if avg > .9:
	   return
	diff = .9 - avg
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * .1)

    def punishDeltaRI(self, delta, LearningRate, i):
        avg = self.predictions[i]
	if avg < .1:
	   return
	diff = avg - .1
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * -1 * LearningRate)

    def rewardDeltaRI(self, delta, LearningRate, i):
	avg = self.predictions[i]
	if avg > .9:
	   return
	diff = .9 - avg
	gradient = Util.sigmoid_prime(diff)
        self.delta = self.delta + (delta * gradient * 1 * LearningRate)

    def compute(self, obs):
	
        self.personalPrediction = Util.sigmoid(np.dot((self.w + self.delta).flatten(), obs))
	self.predictions.append(self.personalPrediction)
	while len(self.predictions) > self.maxSigs:
	    self.predictions.pop(0)
        return self.personalPrediction


