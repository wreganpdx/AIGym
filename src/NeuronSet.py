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

class NeuronSet(object):

    def __init__(self, classes, weights):
        self.weights = weights
	self.structure = (classes, weights)
	self.w = numpy.empty(structure)
        self.seed = time.time()
        print("Creating neuron class: %d seed: %d" %(clas, self.seedActual))
        np.random.seed(seed)  # place a random seed (actually it's just the index, but it works since each perceptron has its own index)


        self.delta =  numpy.empty((classes, weights))
        self.fitness = 0
        self.rekick()
        self.wCopy = numpy.empty((classes, weights))


    def copyWeights(self, index):
        np.copyto(self.wCopy[index], self.w[index])

    def copyWeightSet(self, index):
        np.copyto(self.wCopy, self.w)

    def restoreWeights(self, index):
        np.copyto(self.w[index],self.wCopy[index])

    def restoreWeightSet(self):
        np.copyto(self.w,self.wCopy)


    def backpropPunishSet(self):
        self.w = self.w + self.delta * -1

    def backpropRewardSet(self):
        self.w = self.w + self.delta

    def backpropPunish(self,Index):
        self.w[index] = self.w[index] + self.delta[index] * -1

    def backpropReward(self, Index):
        self.w[index] = self.w[index] + self.delta[index]

    def resetDelta(self):
        self.delta = np.random.rand(self.structure)
        self.delta -= 1.0
        self.delta *= .0001


    def resetDelta(self, index):
        self.delta[index] = np.random.rand(self.weights)
        self.delta[index] -= 1.0
        self.delta[index] *= .0001


    def zeroDelta(self):
        self.delta = self.delta * 0

    def rekick(self):
        self.w = np.random.rand(self.weights)  # set weights
        self.w -= 1.0
        self.w *= .0001
        self.resetDelta()

    def resetFitness(self):
        self.fitness = 0

    def compute(self, obs):
        self.personalPrediction = Util.sigmoid(np.dot(self.w + self.delta, obs.transpose())  )
        self.prediction = self.personalPrediction
        return self.prediction


