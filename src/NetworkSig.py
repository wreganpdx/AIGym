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

class NetworkSig(object):
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
    global results
    results = []
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
	global sigmoids 
	sigmoids = []
	global results
	results = []
	for i in range(len(classes)):
	    t = []
	    sigmoids.append(t)
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
    	NetworkSig.zeroDelta()
    	NetworkSig.resetFitness()   
	gradientCount = 0 
	avgObs = avgObs * 0
	g = '%.4f' %discount
	print("creating gradient w/ discount: %s"% g)


    @staticmethod
    def applyAnnealing(T, steps, rew, scores):
	global gradientCount
	
	annealing = float(T)/float(steps)
        if rew:
            rew = 1
        else:
            rew = -1

        NetworkSig.retrain(rew,annealing)
	NetworkSig.resetFitness()
        NetworkSig.setGradient(T, steps, rew, scores)

	    

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
	global sigmoids
	global results
	result = np.zeros(len(results[0]))
	for i in range(len(results)):
	    result += results[i]
	result = result / len(results)
	totalFitness = 0
	for i in range(len(classes)):
            totalFitness += perceptrons[i].fitness
	avgFitness = float(totalFitness)/len(perceptrons)

	
	flip = (np.random.rand() > .5)
	
	f = 1	
	if flip:
	    f = -1
	randSkip = np.random.rand() - .1
	for i in range(len(classes)):
	    mrRand = np.random.rand()
	    if mrRand > randSkip:
		continue
	    mod = f
	    if perceptrons[i].fitness > avgFitness:
 		mod = mod * -1
	    
	    little = 1 - annealing
	    perceptrons[i].anealSigmoid(annealing, mod, np.average(sigmoids[i]), result)
            for j in range(len(hiddenLayer)):
	        if hiddenLayer[j].getClass() == i:
 	            hiddenLayer[j].delta = (avgObs * mod * little) 

	for i in range(len(sigmoids)):	
	    sigmoids[i] = []
	results = []

    @staticmethod
    def calculateObs(obs, rand = 0):
        global hiddenLayer
        global perceptrons
        global classes
	global lastObs
	global avgObs
	global discount
	global gradientCount
	global sigmoids
	global results
	deltaObs = obs - lastObs
	avgObs = avgObs * discount
	avgObs = avgObs + (1-discount)*deltaObs
        result = np.empty(len(hiddenLayer))
	myRange = len(hiddenLayer)
	
	d = deltaObs.flatten()
	
	d = d.transpose()
        for i in range(myRange):
            result[i] = hiddenLayer[i].compute(d)
	
	lastObs = obs[:]

        scores = np.empty(len(classes))
        bestScore = perceptrons[0].compute(result)
        scores[0] = bestScore
	sigmoids[0].append(bestScore)
        bestIndex = 0
        for i in range(1, len(classes)):
            score = perceptrons[i].compute(result)
	    sigmoids[i].append(score)
            if score > bestScore:
                bestScore = score
                bestIndex = i
	for i in range(0, len(classes)):
	    if len(sigmoids[i]) >= 30:
		sigmoids[i].pop(0)
	results.append(result)
	if len(results) >=30:
	    results.pop(0)
        perceptrons[bestIndex].fitness += 1
	if allowRandom and not np.any(d):
 	    return rand
        return bestIndex


