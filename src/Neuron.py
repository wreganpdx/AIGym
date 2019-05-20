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

class Neuron(object):

    def __init__(self, clas, seed, weights):
        self.weights = weights
        self.clas = clas
        self.seed = seed
        self.seedActual = seed
        print("Creating neuron class: %d seed: %d" %(clas, self.seedActual))
        np.random.seed(seed)  # place a random seed (actually it's just the index, but it works since each perceptron has its own index)
        self.seed = self.clas

        self.delta =  np.random.rand(weights)
        self.delta -= 1.0
        self.delta *= .1

        self.deltaBias = np.random.rand()
        self.deltaBias -= 1.0
        self.deltaBias *= .1
        self.fitness = 0
        self.rekick()
        self.wCopy = self.w[:]


    def getClass(self):
        return self.clas

    def copyWeights(self):
        np.copyto(self.wCopy, self.w)
        self.biasCopy = self.personalBias


    def restoreWeights(self):
        np.copyto(self.w,self.wCopy)
        self.personalBias = self.biasCopy


    def backPropagate(self,reward):
        if reward == 0:
            self.backpropPunish()
        else:
            self.backpropReward()


    def backpropPunish(self):
        self.w = self.w + self.delta * -1
        self.personalBias = self.personalBias + self.deltaBias * -1

    def backpropReward(self):
        self.w = self.w + self.delta
        self.personalBias = self.personalBias + self.deltaBias * 1

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
        self.w -= 1.0
        self.w *= 2
        self.personalBias -= 1.0
        self.personalBias *= 2.0
        self.resetDelta()

    def resetFitness(self):
        self.fitness = 0

    def compute(self, obs, bias=1):
        self.personalPrediction = Util.sigmoid(np.dot(self.w + self.delta, obs.transpose()) + self.personalBias + self.deltaBias )
        self.prediction = self.personalPrediction * bias
        return self.prediction


