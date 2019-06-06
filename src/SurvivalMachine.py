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
import matplotlib.pyplot as plt
from Neuron import Neuron
import sklearn
from sklearn.cluster import KMeans
from BestScores import BestScores
from Util import Util

class SurvivalMachine(object):
    def __init__(self, _nodes, _classes, weights, obs, _metaLayers, trace, _Random):
        self.perceptrons = []
        self.hiddenLayer = []
        self.classes = np.arange(_classes)
        self.lastObs = obs[:]
        self.avgObs = obs[:]
	self.metaLayers = _metaLayers
	self.allowRandom = _Random
        self.space = np.random.rand()
	self.discountSpace = 45
        self.discount = int(self.discountSpace * self.space) + 15
	self.Frames = []
	self.deathFrames = []
	self.classCodes = []
	self.death = []
	self.scoresThisRound = []
	self.roundMax = 10
	self.bestScore = 0
	self.GradientState = 0
	self.TestState = 1
	self.path = ""
    	self.bestAvg = 0
        self.lastAvg = 0
    	self.currentClass = 0
	self.currentDeath = 0
        self.deathOccurances = 0
	self.deathDelta = np.empty(weights)
	self.lastResult = np.empty(weights)
	self.avgDiscount = (self.discountSpace)/2 + 15
   	self.orgDeathOccurances = 0 
	self.states = []
	self.states.append(gradientState(self.GradientState))
	for i in range(len(self.classes)):
	    self.states.append(testState(i+1))
	self.stateMachine = machine(self.GradientState, self.states, self) 
	self.sigmoids = []
	self.annealing = 1
	self.nextChoice = np.arange(_classes)
	np.random.shuffle(self.nextChoice)
	print(self.nextChoice)
	self.currentClass = self.nextChoice[0]
	self.results = []
        for k in range(len(self.classes)):
            p = Neuron(self.classes[k], self.classes[k], _nodes)
            self.perceptrons.append(p)

        for i in range(_nodes):
            h = HiddenNetwork(i * 1000 + i, i %len(self.classes), weights)
            self.hiddenLayer.append(h)
	
    def setPath(self,_path):
	self.path = _path


    def setScoreObj(self,obj):
	self.scoreObj = obj


    def saveWeightsToDisk(self):
	#if bestScore < float(scoreObj.getBest()):
	 #   print("scores too low, not saving")
	  #  return
	p = []
	h = []
	for i in range(len(self.perceptrons)):
	    self.perceptrons[i].w 
	    p.append(self.perceptrons[i].w)
	for k in range(len(self.hiddenLayer)):
	    self.hiddenLayer[k].w 
	    h.append(self.hiddenLayer[k].w)
	avg_s = '%.4f'% self.bestScore
	np.save(self.path+avg_s + "perc", p)
	np.save(self.path+avg_s + "hid", h)
	self.scoreObj.setBest(avg_s)
	self.scoreObj.save()
	print("Saving Weights to disk " + self.path + avg_s)
	


    def saveBestMutationDelta(self):
	p = []
	h = []
	for i in range(len(self.perceptrons)):
	    p.append(self.perceptrons[i].delta)
	for k in range(len(self.hiddenLayer)):
	    h.append(self.hiddenLayer[k].delta)
	np.save(self.path + "dt_perc", p)
	np.save(self.path + "dt_hid", h)


    def loadBestMutationDelta(self):
	print("checking Mutation weights: %s" % self.path)
	if os.path.isfile(self.path + "dt_perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(self.path + "dt_perc.npy")
	h = np.load(self.path + "dt_hid.npy")
	for i in range(len(self.perceptrons)):
	    self.perceptrons[i].delta = p[i]
	a1 = len(self.hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    self.hiddenLayer[k].delta = h[k]


    def loadWeightsFromDisk(self,_path):
	print("checking for weights %s" % _path)
	if os.path.isfile(_path + "perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(_path + "perc.npy")
	h = np.load(_path + "hid.npy")
	for i in range(len(self.perceptrons)):
	    self.perceptrons[i].w = p[i]
	    self.perceptrons[i].zeroDelta()
	    self.perceptrons[i].copyWeights()
	a1 = len(self.hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    self.hiddenLayer[k].w = h[k]
	    self.hiddenLayer[k].zeroDelta()
	    self.hiddenLayer[k].copyWeights()
	

    def setGradient(self):
    	self.space = np.random.rand()
    	self.discount = int(self.discountSpace * self.space) + 15
	self.avgObs = self.avgObs * 0


    def applyAnnealing(self, T, steps, rew, scores):
	self.annealing = float(T)/float(steps)
	return self.stateMachine.update(T, steps, rew, self)

 
    def resetForClass(self):
	self.Frames = []
	self.deathFrames = []
	self.deathOccurances = 0


    def restoreBest(self):
        for i in range(len(self.classes)):
            self.perceptrons[i].restoreWeights()
        for j in range(len(self.hiddenLayer)):
           self.hiddenLayer[j].restoreWeights()
        print("!"),



    def zeroDelta(self):
        for i in range(len(self.classes)):
            self.perceptrons[i].zeroDelta()
        for j in range(len(self.hiddenLayer)):
            self.hiddenLayer[j].zeroDelta()


    def saveBest(self):
        for i in range(len(self.classes)):
            self.perceptrons[i].backpropReward()
            self.perceptrons[i].copyWeights()
        for j in range(len(self.hiddenLayer)):
            self.hiddenLayer[j].backpropReward()
            self.hiddenLayer[j].copyWeights()

        print("*"),
	self.saveWeightsToDisk()



    def createDeathFrame(self):
	length = len(self.Frames)
	if length == 0:
	    length = 1
	self.avgObs = self.avgObs * 0
	for i in range(len(self.Frames)):
	    self.avgObs = self.avgObs + self.Frames[i]
	self.avgObs = self.avgObs/length
	self.deathFrames.append(self.avgObs[:])	


    def calculateObs(self,obs, rand = 0):
	unique = np.unique(obs)
	originalDelta = obs - self.lastObs
	deltaObs = originalDelta[:]
    	deltaObs = np.abs(deltaObs)
    	self.Frames.append(self.avgObs[:])
    	if len(self.Frames) >= self.discount:
	    self.Frames.pop(0)
	
	    
    	self.avgObs = self.avgObs * .8 + deltaObs * .2 
		
	    
        self.lastResult = np.empty(len(self.hiddenLayer))
	myRange = len(self.hiddenLayer) 
	
	d = originalDelta.flatten()
	
	d = d.transpose()
        for i in range(myRange):
            self.lastResult[i] = self.hiddenLayer[i].compute(d)

	self.lastObs = obs[:]

        self.scores = np.empty(len(self.classes))
        b = self.perceptrons[0].compute(self.lastResult)
        self.scores[0] = b
        bestIndex = 0
	if self.currentClass == 0:
	    self.sigmoids.append(b)
        for i in range(1, len(self.classes)):
            score = self.perceptrons[i].compute(self.lastResult)
	    if self.currentClass == i:
		self.sigmoids.append(score)
            if score > b:
                b = score
                bestIndex = i
	if len(self.sigmoids) >= self.discount:
	    self.sigmoids.pop(0)
	
        self.perceptrons[bestIndex].fitness += 1
	self.results.append(self.lastResult)
	if len(self.results) >= self.discount:
	    self.results.pop(0)
	if self.allowRandom and not np.any(d):
 	    return rand
        return bestIndex



    def setClassLabels(self,labels):
	self.classLabels = labels


class gradientState(object):
    def __init__(self, id):
	self.id = id
	self.done = False
	self.scoresThisRound = []
    	return
    
    def enter(self, net):
	self.done = False
	self.scoresThisRound = []
	return

    def exit(self, net):
	net.death = net.deathFrames[0]
	net.deathFrames = []
	success = self.scoresThisRound[0]
	net.bestScore = success
	net.lastAvg = success
	success_s = '%.4f' % success
	print("Gradient EV: %s" % (success_s))
	print("---exit gradient")
	return

    def update(self,T, steps, rew, net):
	self.scoresThisRound.append(rew)
        net.createDeathFrame()
	self.done = self.validateScore(T, steps, rew, net)
	if !self.done:
	    _path = net.path
	    b = '%.4f' % float(net.scoreObj.getBest())
	    net.loadWeightsFromDisk(self,_path + b)
	    self.done = True
	    
    
    def validateScore(self, T, steps, rew, net):
	if self.scoresThisRound[0] >= net.scoreObj.getBest():
	    return True
	rand = np.random.rand()
	if rand < float(T)/float(steps):
	    return True
	return False
    def execute(self, net):
	if self.done:
	    return net.TestState
	return self.id

class testState(object):
    def __init__(self, id):
	self.id = id
	self.done = False
	self.scoresThisRound = []
	self.deathOccurances = 0
	return

    def enter(self,net):
	self.done = False
	self.deathOccurances = 0
	self.scoresThisRound = []
	return


    def exit(self,net):
	success = float(np.count_nonzero(self.scoresThisRound))/float(len(self.scoresThisRound))
	avg = np.average(self.scoresThisRound)
	print(self.scoresThisRound)
	if success == 1:
	    success = avg
	
	avg_s = '%.4f' % avg
	success_s = '%.4f' % success
	
	if net.bestScore < success:
            net.saveBestMutationDelta()
	    net.bestScore = success
	else:
	    print("EV for reward too low, rejecting new weights")
	
	net.resetForClass()
	net.zeroDelta()
	print("%s Results: AVG: %s, EV: %s" % (net.classLabels[net.currentClass], avg_s,success_s))
	
	
	net.nextChoice = np.delete(net.nextChoice,0,0)
	if len(net.nextChoice) == 0:
	    net.loadBestMutationDelta()
	    net.saveBest()
	    net.nextChoice = np.arange(len(net.classes))
	    np.random.shuffle(net.nextChoice)
	    print("new order: "),
            print(net.nextChoice)
	    print("---exit training")
        net.currentClass = net.nextChoice[0]
        net.discount = int(net.discountSpace * net.space) + 15

    def update(self,T, steps, rew, net):

	self.scoresThisRound.append(rew)
	if len(self.scoresThisRound) >= net.roundMax:
	    self.done = True
	    return
        net.createDeathFrame()
	self.trainClassifier(T, steps, rew, net)
	

    
    def trainClassifier(self, T, steps, rew, net):
	c1 = net.currentClass
	perceptrons = net.perceptrons
	hiddenLayer = net.hiddenLayer
	dFrames = net.deathFrames
	fLast = len(net.deathFrames) -1
	for j in range(len(hiddenLayer)):
	    if hiddenLayer[j].getClass() == c1:
		hiddenLayer[j].rewardDelta(dFrames[fLast])
	    else:
		hiddenLayer[j].punishDelta(dFrames[fLast])
	result = np.zeros(net.results[0].size)
	for i in range(len(net.results)):
	    result += net.results[i]
	result = result / len(net.results)
	for i in range(len(perceptrons)):
	    if perceptrons[i] == c1:
		perceptrons[i].rewardDelta(result, np.average(net.sigmoids[i]))
	    else:
		perceptrons[i].punishDelta(result, np.average(net.sigmoids[i]))
    
    def execute(self, net):
	if self.done and self.id == len(net.classes):
	    return net.GradientState
        if self.done:
	    return self.id +1
	return self.id

class machine(object):
    def __init__(self, currentState, states, net):
	
	self.states = states
        self.currentState = states[currentState]
        self.currentState.enter(net)

    def update(self,T, steps, rew, net):
	self.currentState.update(T, steps, rew,net)
	next = self.currentState.execute(net)
	next = self.states[next]
	if next.id != self.currentState.id:
	    self.currentState.exit(net)
	    next.enter(net)
	    self.currentState = next
	    if self.currentState.id == net.GradientState: 
            	return (True,True)
	    return (True,False)
        return (False, False)
	
