"""
William C Regan
Student ID: 954681718
Machine Learning
Portland State University
Final Project
"""
import os
import numpy as np
from HiddenNetwork import  HiddenNetwork
from Util import Util

from Neuron import Neuron

class NetworkObs(object):
    global perceptrons
    perceptrons = []
    global hiddenLayer
    perceptrons = []
    global classes
    classes = []
    global lastObs
    lastObs = []
    global avgObs
    avgObs = []
    global metaLayers
    metaLayers = 1
    global trace
    trace = "obs"
    global allowRandom
    allowRandom = False
    global discount
    discount = .75
    global gradientCount
    gradientCount = 0
    def __init__(self, _nodes, _classes, weights, obs, _metaLayers, trace, _Random):
        global perceptrons
        perceptrons = []
        global hiddenLayer
        hiddenLayer = []
        global classes
        classes = np.arange(_classes)
        global lastObs
        lastObs = obs[:]
	global avgObs
        avgObs = obs[:]
	global metaLayers
	metaLayers = _metaLayers
        global allowRandom
	allowRandom = _Random
	global discount
        discount = .75
	global gradientCount
	gradientCount = 0
	_nodes = _nodes * _metaLayers
        for k in range(len(classes)):
            p = Neuron(classes[k], classes[k], _nodes)  # important to set perceptrons here for each new learning rate
            perceptrons.append(p)

        for i in range(_nodes):
            h = HiddenNetwork(i * 1000 + i, i %len(classes), weights)  # important to set perceptrons here for each new learning rate
            hiddenLayer.append(h)

    @staticmethod 
    def saveWeightsToDisk(path):
	global perceptrons
	global hiddenLayer
	p = []
	h = []
	for i in range(len(perceptrons)):
	    p.append(perceptrons[i].w)
	for k in range(len(hiddenLayer)):
	    h.append(hiddenLayer[k].w)
	np.save(path + "ObsPerc", p)
	np.save(path + "ObsHid", h)

    @staticmethod 
    def loadWeightsFromDisk(path):
	global perceptrons
	global hiddenLayer
	print("checking for weights %s" % path)
	if os.path.isfile(path + "ObsPerc.npy") != True:
	    return
	print("loading weights")
	p = np.load(path + "ObsPerc.npy")
	h = np.load(path + "ObsHid.npy")
	for i in range(len(perceptrons)):
	    perceptrons[i].w = p[i]
	    perceptrons[i].zeroDelta()
	    perceptrons[i].copyWeights()
	a1 = len(hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    hiddenLayer[k].w = h[k]
	    hiddenLayer[k].zeroDelta()
	    hiddenLayer[k].copyWeights()
	
    @staticmethod
    def setGradient(T, steps, rew, scores):
	global discount
	global gradientCount
	global avgObs
    	space = np.random.rand()
    	space = space * .6
    	discount = 1 - space
    	NetworkObs.zeroDelta()
    	NetworkObs.resetFitness()   
	gradientCount = 0 
	avgObs = avgObs * 0
	g = '%.4f' %discount
	print("creating gradient w/ discount: %s"% g)


    @staticmethod
    def applyAnnealing(T, steps, rew, scores):
	global gradientCount
	if gradientCount > 5:
	    NetworkObs.setGradient(T, steps, rew, scores)
	    return
        else:
	    gradientCount += 1
	annealing = float(T)/float(steps)
        if rew:
            rew = 1
        else:
            rew = -1
        rand = np.random.rand()
        if rand > annealing:
            NetworkObs.retrain(rew,annealing)
        else:
            NetworkObs.restoreBest()
	NetworkObs.resetFitness()

    @staticmethod
    def localMax(scores):
	if len(scores) == 0:
	    return False
	biggest = np.argmax(scores)
	if len(scores) - 15 > biggest:
	    return True
	return False

    @staticmethod
    def restoreBest():
        global perceptrons
        global hiddenLayer
        for i in range(len(classes)):
            perceptrons[i].restoreWeights()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].restoreWeights()
        print("!"),

    @staticmethod
    def resetFitness():
        global perceptrons
	global avgObs
        for i in range(len(classes)):
            perceptrons[i].fitness = 0
    @staticmethod
    def zeroDelta():
        global perceptrons
        global hiddenLayer
        for i in range(len(classes)):
            perceptrons[i].zeroDelta()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].zeroDelta()

    @staticmethod
    def saveBest():
        global perceptrons
        global hiddenLayer
	global discount
	discount = .75
        for i in range(len(classes)):
            perceptrons[i].backpropReward()
            perceptrons[i].copyWeights()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].backpropReward()
            hiddenLayer[j].copyWeights()

        print("*"),




    @staticmethod
    def retrain(reward, annealing):
	global classes
        global perceptrons
        global hiddenLayer
	global avgObs
	totalFitness = 0
	for i in range(len(classes)):
            totalFitness += perceptrons[i].fitness
	avgFitness = float(totalFitness)/len(perceptrons)

	
	flip = (np.random.rand() > .5)
	
	f = 1	
	if flip:
	    f = -1
	
	for i in range(len(classes)):
	    mod = f
	    if perceptrons[i].fitness > avgFitness:
 		mod = mod * -1
	    mrRand = np.random.rand()
	    if mrRand < perceptrons[i].fitness:
	    	perceptrons[i].anealDelta(annealing, mod)
	    else:
		mrRand *= 2
            for j in range(len(hiddenLayer)):
	        if hiddenLayer[j].getClass() == i:
 	            hiddenLayer[j].delta = (mrRand * avgObs * mod) 



    @staticmethod
    def calculateObs(obs, rand = 0):
        global hiddenLayer
        global perceptrons
        global classes
	global lastObs
	global avgObs
	global metaLayers
	global discount
	global gradientCount
	deltaObs = obs - lastObs
	if gradientCount == 0:
	    avgObs = avgObs * discount
	    avgObs = avgObs + (1-discount)*deltaObs
        result = np.empty(len(hiddenLayer))
	myRange = len(hiddenLayer) / metaLayers
	
	d = deltaObs.flatten()
	if allowRandom and not np.any(d):
 	    return rand
	d = d.transpose()


    	o = obs.flatten().transpose()
    	for i in range(0, myRange):
            result[i] = hiddenLayer[i].compute(o)
	lastObs = obs[:]

        scores = np.empty(len(classes))
        bestScore = perceptrons[0].compute(result)
        scores[0] = bestScore
        bestIndex = 0
        for i in range(1, len(classes)):
            score = perceptrons[i].compute(result)
            if score > bestScore:
                bestScore = score
                bestIndex = i

        perceptrons[bestIndex].fitness += 1
        return bestIndex


    @staticmethod
    def getPerceptrons():
        return perceptrons



    @staticmethod
    def numNeurons():
        return len(perceptrons)

