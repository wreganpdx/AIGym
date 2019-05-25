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

class Network(object):
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
        global allowRandom
	allowRandom = _Random
	metaLayers = _metaLayers
	
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
	a1 = len(hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    hiddenLayer[k].w = h[k]
	    hiddenLayer[k].zeroDelta()
	    
    @staticmethod
    def applyAnnealing(T, steps, rew):
	annealing = float(T)/float(steps)
        if rew:
            rew = 1
        else:
            rew = -1
        rand = np.random.rand();
        if rand > annealing:
            Network.retrain(rew,annealing)
        else:
            rand = np.random.rand();
            if rand > annealing and rew < 1:
                Network.rekick()
            else:
                Network.restoreBest()
	Network.resetFitness()

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
        for i in range(len(classes)):
            perceptrons[i].backpropReward()
            perceptrons[i].copyWeights()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].backpropReward()
            hiddenLayer[j].copyWeights()

        print("*"),

    @staticmethod
    def rekick():
        global perceptrons
        global hiddenLayer
        for i in range(len(classes)):
            perceptrons[i].rekick()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].rekick()


    @staticmethod
    def retrain3(reward):
	global classes
        global perceptrons
        global hiddenLayer
	totalFitness = 0
	for i in range(len(classes)):
            totalFitness += perceptrons[i].fitness
	avgFitness = float(totalFitness)/len(perceptrons)
	
	for i in range(len(classes)):
            perceptrons[i].backPropagate(reward)
	    mod = 1
	    if perceptrons[i].fitness > avgFitness:
 		mod = -1
		perceptrons[i].backPropagate(reward)
                for j in range(len(hiddenLayer)):
		    if hiddenLayer[j].getClass() == i:
	 	        hiddenLayer[j].delta = hiddenLayer[j].delta * mod 
                        hiddenLayer[j].backPropagate(reward)

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


        #lottery = np.random.rand()
     #   lottery *= avgFitness
	
	flip = (np.random.rand() > .5)
	mrRand = np.random.rand()
	f = 1	
	if flip:
	    f = -1
	
	for i in range(len(classes)):
	    mod = f
	    if perceptrons[i].fitness > avgFitness:
 		mod = mod * -1
	    perceptrons[i].anealDelta(annealing, mod)

            for j in range(len(hiddenLayer)):
	        if hiddenLayer[j].getClass() == i:
 	            hiddenLayer[j].delta = (mrRand * avgObs * mod) 

    @staticmethod
    def retrain2(reward):
        global classes
        global perceptrons
        global hiddenLayer
        totalFitness = 0
        for i in range(len(classes)):
            totalFitness += perceptrons[i].fitness

        indiBonus = totalFitness/len(classes)
        totalFitness *= 2

        lottery = np.random.rand()
        lottery *= totalFitness
        fit = 0
        for i in range(len(classes)):
            fit += (perceptrons[i].fitness + indiBonus)
            if lottery < fit:
                perceptrons[i].backPropagate(reward)
                for j in range(len(hiddenLayer)):
                    if hiddenLayer[j].getClass() == i:
			#hiddenLayer[j].
                        hiddenLayer[j].backPropagate(reward)
		break

    @staticmethod
    def resetDelta():
        global perceptrons
        global hiddenLayer
        for i in range(len(perceptrons)):
            perceptrons[i].resetDelta()
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].resetDelta()


    @staticmethod
    def calculateObs(obs, rand = 0):
        global hiddenLayer
        global perceptrons
        global classes
	global lastObs
	global avgObs
	global metaLayers
	
	deltaObs = obs - lastObs
	avgObs = avgObs * .75
	avgObs = avgObs + .25*deltaObs
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

