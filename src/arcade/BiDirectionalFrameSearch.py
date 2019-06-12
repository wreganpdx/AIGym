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

class BiDirectionalFrameSearch(object):
    def __init__(self, _nodes, _classes, weights, obs, _metaLayers, trace, _Random):
        self.perceptrons = []
        self.hiddenLayer = []
        self.classes = np.arange(_classes)
        self.lastObs = obs[:]
	self.allowRandom = _Random
        self.discount = 75
	self.Frames = []
	self.classCodes = []
	self.roundMax = 15
	self.bestScore = 0
	self.GradientState = 0
	self.TestState = 1
	self.path = ""
    	self.currentClass = 0
	self.deathDelta = np.empty(weights)
	self.lastResult = np.empty(weights)
	self.states = []
	self.states.append(gradientState(self.GradientState))
	for i in range(len(self.classes)):
	    self.states.append(testState(i+1))
	self.stateMachine = machine(self.GradientState, self.states, self) 
	self.sigmoids = []
	for i in range(len(self.classes)):
	    self.sigmoids.append([])
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
	    h.maxSigs = self.discount
	self.winners = []

    def setPath(self,_path):
	self.path = _path


    def setScoreObj(self,obj):
	self.scoreObj = obj

	
    def preSaveWeightsToDisk(self, score):
	p = []
	h = []
	for i in range(len(self.perceptrons)):
	    p.append(self.perceptrons[i].w + self.perceptrons[i].delta)
	for k in range(len(self.hiddenLayer)):
	    h.append(self.hiddenLayer[k].w + self.hiddenLayer[k].delta)
	avg_s = '%.4f'% score
	np.save(self.path+avg_s + "perc", p)
	np.save(self.path+avg_s + "hid", h)
	self.scoreObj.setBest(avg_s)
	self.scoreObj.save()
	print("Pre-Saving Weights to disk " + self.path + avg_s)



    def loadWeightsFromDisk(self,_path):
	print("checking for weights %s" % _path)
	if os.path.isfile(_path + "perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(_path + "perc.npy")
	h = np.load(_path + "hid.npy")
	self.zeroDelta()
	for i in range(len(self.perceptrons)):
	    self.perceptrons[i].w = p[i]
	    self.perceptrons[i].copyWeights()
	a1 = len(self.hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    self.hiddenLayer[k].w = h[k]
	    self.hiddenLayer[k].copyWeights()


    def applyAnnealing(self, T, steps, rew, scores):
	return self.stateMachine.update(T, steps, rew, self)



    def zeroDelta(self):
        for i in range(len(self.classes)):
            self.perceptrons[i].zeroDelta()
        for j in range(len(self.hiddenLayer)):
            self.hiddenLayer[j].zeroDelta()




    def calculateObs(self,obs, rand = 0):
	deltaObs = np.abs(obs - self.lastObs)
	self.lastObs = obs[:]
    	self.Frames.append(deltaObs)
    	while len(self.Frames) > self.discount:
	    self.Frames.pop(0)
	
	    
		
	    
        self.lastResult = np.empty(len(self.hiddenLayer))
	#self.lastResult[len(self.perceptrons)] = 1
	myRange = len(self.hiddenLayer) 
	
	d = deltaObs.flatten().transpose()
	
        for i in range(myRange):
            self.lastResult[i] = self.hiddenLayer[i].compute(d)



        b = self.perceptrons[0].compute(self.lastResult)
        bestIndex = 0
	self.sigmoids[0].append(b)
	while len(self.sigmoids[0]) > self.discount:
	    self.sigmoids[0].pop(0)
	
        for i in range(1, len(self.classes)):
            score = self.perceptrons[i].compute(self.lastResult)
	    self.sigmoids[i].append(score)
            if score > b:
                b = score
                bestIndex = i
	    while len(self.sigmoids[i]) > self.discount:
	    	self.sigmoids[i].pop(0)
	
	self.results.append(self.lastResult)
	while len(self.results) > self.discount:
	    self.results.pop(0)
	if self.allowRandom and not np.any(d):
 	    return rand
	z = np.random.rand()
	if z < .01:
	    return rand
	self.winners.append(bestIndex)
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
	b = '%.4f' % net.scoreObj.getBest()
	net.loadWeightsFromDisk(net.path + b)
	net.zeroDelta()
	net.winners = []
	return

    def exit(self, net):
	success = self.score
	net.bestScore = success
	net.lastAvg = success
	success_s = '%.4f' % success
	print("Gradient EV: %s" % (success_s))
	#print(net.winners)
	print("---exit gradient")
	net.scoreObj.confirmBest(success)
	return

    def update(self,T, steps, rew, net):
	self.score = rew
	self.done = True
	    
    def execute(self, net):
	if self.done:
	    return net.TestState
	return self.id

class testState(object):
    def __init__(self, id):
	self.id = id
	
	return

    def enter(self,net):
	self.done = False
	self.scoresThisRound = []
	self.actionsThisRound = []
	self.trainedThisRound = []

        self.currIndex = 0
	rand = np.random.rand()
	self.startIndex = int(rand * net.discount)
	if self.startIndex < 5:
	    self.startIndex = 5
	if self.startIndex > net.discount-5:
	    self.startIndex = net.discount -5
	#print(net.winners)
	return


    def exit(self,net):
	success = float(np.count_nonzero(self.scoresThisRound))/float(len(self.scoresThisRound))
	avg = np.average(self.scoresThisRound)
	print("****")
	print("Scores: "),
	print(self.scoresThisRound)
	print("Actions: "),
	print(self.actionsThisRound)
	print("Trained: "),
	print(self.trainedThisRound)
	if success == 1:
	    success = avg
	
	avg_s = '%.4f' % avg
	success_s = '%.4f' % success
	
	net.Frames = []
	net.zeroDelta()
	print("%s Results: AVG: %s, EV: %s" % (net.classLabels[net.currentClass], avg_s,success_s))
	
	
	net.nextChoice = np.delete(net.nextChoice,0,0)
	if len(net.nextChoice) == 0:
	    net.nextChoice = np.arange(len(net.classes))
	    np.random.shuffle(net.nextChoice)
	    print("new order: "),
            print(net.nextChoice),
	    print("---exit training")
	#print(net.winners)
        net.currentClass = net.nextChoice[0]
	self.scoresThisRound = []
	self.actionsThisRound = []
	self.trainedThisRound = []
	
    def countActions(self, net):
	sigs = net.sigmoids
	actions = len(sigs[0])
	for i in range(len(sigs[0])):
	    score = sigs[net.currentClass][i]
	    for j in range(len(sigs)):
		if j == net.currentClass:
		    continue
		if sigs[j][i] >= score:
		    actions -= 1
		    break
	self.actionsThisRound.append(actions)
	return actions

    def update(self,T, steps, rew, net):
	
	self.scoresThisRound.append(rew)
	if len(self.scoresThisRound) >= net.roundMax:
	    self.done = True
	    return

	if self.countActions(net) == net.discount:
	    self.done = True
	    return

	if rew > net.scoreObj.getBest():
	    net.preSaveWeightsToDisk(rew)
	if rew < net.bestScore:
	    self.done = True
	else:
	    self.trainClassifier(T, steps, rew, net)

	if self.done != True:
	    net.winners = []
	
	

    
    def trainClassifier(self, T, steps, rew, net):
	self.trainClassifierResult(T, steps, rew, net)
	cur = self.startIndex
	numTrained = 0
	rightIndex = self.currIndex
	while cur < net.discount and cur <= self.startIndex + self.currIndex:
	    trained = self.trainClassifierFrame(T, steps, rew, net, cur)
	    if trained:
		numTrained += 1
	    else:
		rightIndex += 1
	    cur +=1
	
	cur = self.startIndex -1
	leftIndex = self.currIndex
	while cur >= 0 and cur > self.startIndex - self.currIndex:
	    trained = self.trainClassifierFrame(T, steps, rew, net, cur)
	    if trained:
		numTrained += 1
	    else:
		leftIndex -= 1
	    cur -=1
	self.trainedThisRound.append(numTrained)
	self.currIndex += 1
	
    def trainClassifierResult(self, T, steps, rew, net):
	trained = False
	sigs = net.sigmoids

	c1 = net.currentClass
	perceptrons = net.perceptrons
	start = self.startIndex - self.currIndex
	end = self.startIndex + self.currIndex
	if start < 0:
	    start = 0
	if end > net.discount:
	    end = net.discount
	result = np.sum( net.results[start:end], axis=0)
	result = result / len(net.results)
	

    	test = np.average(sigs[net.currentClass])

    	omitList = []
    	for k in range(len(perceptrons)):
            if np.average(sigs[k]) < test and k != (net.currentClass):
	    	omitList.append(k)
    	if len(omitList) == (len(perceptrons) -1):
	    print("-"),
	    return trained
	

    	LearningRate = np.random.rand() *.001  + .001
    	for k in range(len(perceptrons)):
	    if np.isin(k, omitList):
                continue
	    sig = np.average(sigs[k])
	    rand = np.random.rand()
    	    if perceptrons[k] == c1:
		perceptrons[k].zeroDelta()
	    	perceptrons[k].rewardDeltaRC(result, sig, LearningRate)
    	    else:
		perceptrons[k].zeroDelta()
		perceptrons[k].punishDeltaRC(result, sig, LearningRate)
    
	trained = True
	return trained
    def trainClassifierFrame(self, T, steps, rew, net, i):
	trained = False
	sigs = net.sigmoids
	if i >= len(sigs[0]):
            return trained
	if i < 0:
	    return trained

	c1 = net.currentClass
	perceptrons = net.perceptrons
	hiddenLayer = net.hiddenLayer
	frame = net.Frames[i]
	result = net.results[i]
	

    	test = sigs[net.currentClass][i]

    	omitList = []
    	for k in range(len(perceptrons)):
            if sigs[k][i] < test and k != (net.currentClass):
	    	omitList.append(k)
    	if len(omitList) == (len(perceptrons) -1):
	    print("-"),
	    return trained
	

    	LearningRate = np.random.rand() *.01  + .01
    	for k in range(len(perceptrons)):
    	    if np.isin(k, omitList):
                continue
	    if hiddenLayer[k].getClass() == c1:
	    	hiddenLayer[k].rewardDeltaRI(frame, LearningRate,i)
    	    else:
	    	hiddenLayer[k].punishDeltaRI(frame, LearningRate,i)
	trained = True
	return trained
	    
    
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

    def update(self,T, steps, rew, net):
	self.currentState.update(T, steps, rew,net)
	next = self.currentState.execute(net)
	next = self.states[next]
	if next.id != self.currentState.id:
	    self.currentState.exit(net)
	    next.enter(net)
	    self.currentState = next

	
