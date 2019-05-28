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

class NetworkNorm(object):
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
	np.save(path + "perc", p)
	np.save(path + "hid", h)

    @staticmethod 
    def loadWeightsFromDisk(path):
	global perceptrons
	global hiddenLayer
	print("checking for weights %s" % path)
	if os.path.isfile(path + "perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(path + "perc.npy")
	h = np.load(path + "hid.npy")
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
    	discount = 1 - space
    	NetworkNorm.zeroDelta()
    	NetworkNorm.resetFitness()   
	gradientCount = 0 
	avgObs = avgObs * 0
	g = '%.4f' %discount
	print("creating gradient w/ discount: %s"% g)


    @staticmethod
    def applyAnnealing(T, steps, rew, scores):
	global gradientCount
	if gradientCount > 5:
	    NetworkNorm.setGradient(T, steps, rew, scores)
	    return
        else:
	    gradientCount += 1
	annealing = float(T)/float(steps)
        if rew:
            rew = 1
        else:
            rew = -1

        NetworkNorm.retrain(rew,annealing)
	NetworkNorm.resetFitness()

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
	    little = 1 - annealing
	    if mrRand > annealing:
		perceptrons[i].anealDelta(annealing, reward)
            for j in range(len(hiddenLayer)):
	        if hiddenLayer[j].getClass() == i:
 	            hiddenLayer[j].delta = (avgObs * mod * little) 



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
	deltaObs = np.square(deltaObs)
	if gradientCount == 0:
	    avgObs = avgObs * discount
	    avgObs = avgObs + (1-discount)*deltaObs
        result = np.empty(len(hiddenLayer))
	myRange = len(hiddenLayer) / metaLayers
	
	d = deltaObs.flatten()
	if allowRandom and not np.any(d):
 	    return rand
	d = d.transpose()
        for i in range(myRange):
            result[i] = hiddenLayer[i].compute(d)
	if metaLayers >= 2:
	    o = obs.flatten().transpose()
	    for i in range(myRange, myRange*2):
                result[i] = hiddenLayer[i].compute(o)
	if metaLayers >= 3:
	    a = avgObs.flatten().transpose()
	    for i in range(myRange*2, myRange*3):
                result[i] = hiddenLayer[i].compute(a)
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

