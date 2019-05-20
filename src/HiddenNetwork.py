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
        np.random.seed(seed)  # place a random seed (actually it's just the index, but it works since each perceptron has its own index)

        self.rekick()
        self.delta = np.random.rand(weights)
        self.delta -= 1.0
        self.delta *= .1

        self.deltaBias = np.random.rand();
        self.deltaBias -= 1.0
        self.deltaBias *= .1
        self.wCopy = self.w[:]
        self.copyWeights()

    def getClass(self):
        return self.clas

    def copyWeights(self):
        np.copyto(self.wCopy, self.w)
        self.biasCopy = self.personalBias


    def restoreWeights(self):
        np.copyto(self.w,self.wCopy)
        self.personalBias = self.biasCopy

        #print(self.w, self.personalBias),


    def backPropagate(self, reward):
        if reward == 0:
            self.backpropPunish()
        else:
            self.backpropReward()

    def resetDelta(self):
        self.delta = np.random.rand(self.weights)
        self.delta -= 1.0
        self.delta *= .1
        self.deltaBias = np.random.rand();
        self.deltaBias -= 1.0
        self.deltaBias *= .1

    def rekick(self):
        self.w = np.random.rand(self.weights)  # set weights
        self.personalBias = random.random()
        self.personalBias += -1
        self.personalBias *= .5
        for i in range(len(self.w)):
            self.w[i] -= 1  # adjust weights so we have positive and negative weights
            self.w[i] *= .5

        self.resetDelta()


    def backpropPunish(self):
        self.w = self.w +  self.delta * -1
        self.personalBias = self.personalBias - self.deltaBias
        self.resetDelta()

    def backpropReward(self):
        self.w = self.w + self.delta
        self.personalBias = self.personalBias + self.deltaBias
        self.resetDelta()


    def compute(self, obs):
        self.personalPrediction = Util.sigmoid(np.dot(self.w + self.delta, obs.transpose()) + self.personalBias + self.deltaBias)
        return self.personalPrediction


