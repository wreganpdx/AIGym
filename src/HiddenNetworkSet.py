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
class HiddenNetwork(object):

    def __init__(self, seed, clas, weights):
        self.weights = weights
        self.clas = clas
        self.seed = seed
        self.seedActual = seed
        print("Creating hidden neuron class: %d seed:"% seed)
        np.random.seed(seed)  

        self.rekick()
	self.resetDelta()
        self.wCopy = self.w[:]
        self.copyWeights()

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
        self.w *= .0001
        self.resetDelta()


    def backpropPunish(self):
        self.w = self.w +  self.delta * -1
        self.resetDelta()

    def backpropReward(self):
        self.w = self.w + self.delta
        self.resetDelta()


    def compute(self, obs):
	
        self.personalPrediction = Util.sigmoid(np.dot((self.w + self.delta).flatten(), obs))
        return self.personalPrediction


