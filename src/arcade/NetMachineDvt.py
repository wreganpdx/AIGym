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

class NetMachineDvt(object):
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
	self.roundMax = 100
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
	self.states.append(testState(self.TestState))
	self.stateMachine = machine(self.GradientState, self.states, self) 
	self.sigmoids = []
	self.annealing = 1
	self.nextChoice = np.arange(_classes)
	np.random.shuffle(self.nextChoice)
	print(self.nextChoice)
	self.currentClass = self.nextChoice[0]
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


    def loadBestMutationDelta(self,omit):
	print("checking Mutation weights: %s omiting: %d" % (self.path, omit))
	if os.path.isfile(self.path + "dt_perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(self.path + "dt_perc.npy")
	h = np.load(self.path + "dt_hid.npy")
	for i in range(len(self.perceptrons)):
	    if i == omit:
		continue;
	    self.perceptrons[i].delta = p[i]
	a1 = len(self.hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    if k == omit:
		continue;
	    self.hiddenLayer[k].delta = h[k]


    def loadWeightsFromDisk(self,_path):
	path = _path
	print("checking for weights %s" % self.path)
	if os.path.isfile(self.path + "perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(self.path + "perc.npy")
	h = np.load(self.path + "hid.npy")
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
	self.bestScore = 0
	self.bestAvg = 0
	self.deathOccurances = 0

    def resetVision(self):
	self.Frames = []
	self.deathFrames = []
	self.bestScore = 0
	self.bestAvg = 0
	self.deathOccurances = 0

	

    def clusterdeaths(self):
	_deaths = []
	frequencies = np.zeros(len(self.classes))
	for i in range(len(self.deathFrames)):
	    _deaths.append(self.deathFrames[i].flatten())
	shape = self.deathFrames[0].shape
	deathFrames = []
	self.kmeans = KMeans(n_clusters=len(self.classes)).fit(_deaths)
  
	predictions = self.kmeans.predict(_deaths)
	for i in range(len(predictions)):
	    frequencies[predictions[i]] = frequencies[predictions[i]] + 1
    #frequencies[index] = frequencies[index] + 1
	centroids = []
	for i in range(len(self.kmeans.cluster_centers_)):
	    centroids.append(np.reshape(self.kmeans.cluster_centers_[i], shape))
	print("Death Clusters Created")
	
	self.orgDeathOccurances = np.max(frequencies)
	self.currentDeath = np.argmax(frequencies)
	#for i in range(len(centroids)):
	 #   plt.imshow(centroids[i])
	  #  plt.figure()
	   # plt.show()
	self.death = np.array(centroids[self.currentDeath], copy=True)


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
	if self.allowRandom and not np.any(d):
 	    return rand
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
	if len(self.sigmoids) > self.avgDiscount:
	    self.sigmoids.pop(0)
        self.perceptrons[bestIndex].fitness += 1

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
	net.clusterdeaths()
	net.deathFrames = []

	success = float(np.count_nonzero(self.scoresThisRound))/float(len(self.scoresThisRound))
	avg = np.average(self.scoresThisRound)
	print(self.scoresThisRound)
	if success == 1:
	    success = avg
	net.bestScore = success
	net.lastAvg = success
	avg_s = '%.4f' % avg
	success_s = '%.4f' % success
	print("Gradient avg: %s, EV: %s" % (avg_s,success_s))
	print("---exit gradient")
	return

    def update(self,T, steps, rew, net):
	self.scoresThisRound.append(rew)
        net.createDeathFrame()
	net.setGradient()
	if len(net.deathFrames) >= net.roundMax:
	   self.done = True
    
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

    def trainOne(c1, net):
	for j in range(len(hiddenLayer)):
            if net.hiddenLayer[j].getClass() == c1:
    	        net.hiddenLayer[j].delta = net.death 

    def exit(self,net):
	


	success = float(np.count_nonzero(self.scoresThisRound))/float(len(self.scoresThisRound))
	avg = np.average(self.scoresThisRound)
	print(self.scoresThisRound)
	if success == 1:
	    success = avg
	net.bestScore = success
	avg_s = '%.4f' % avg
	success_s = '%.4f' % success


	self.gradientDelta(net)
	
	if net.lastAvg <= success * 1.05 or net.lastAvg == success:
             net.saveBest()
	else:
	     print("EV for reward too low, rejecting new weights")
	
	net.resetVision()
	net.resetForClass()
	net.zeroDelta()
	print("%s Results: AVG: %s, EV: %s" % (net.classLabels[net.currentClass], avg_s,success_s))
	
	
	net.nextChoice = np.delete(net.nextChoice,0,0)
	if len(net.nextChoice) == 0:
	   
	    
	    net.nextChoice = np.arange(len(net.classes))
	    np.random.shuffle(net.nextChoice)
	    print("new order: "),
            print(net.nextChoice)
	print("---exit training")
        net.currentClass = net.nextChoice[0]

    def update(self,T, steps, rew, net):
	self.scoresThisRound.append(rew)
        net.createDeathFrame()
	_death = []
	curFrame = net.deathFrames[len(net.deathFrames)-1]
	_death.append(curFrame.flatten())
	d = net.kmeans.predict(_death)
	d = d[0]
	if d == net.currentDeath:
	    self.deathOccurances += 1
	    self.continueTraining(np.average(net.sigmoids),curFrame, net)
	else:
	    net.sigmoids.pop()
	if len(self.scoresThisRound) >= net.roundMax:

	    self.done = True

    def continueTraining(self, sigmoid, death, net):
	if sigmoid > .9:
	   return

	gradient = Util.sigmoid_prime(sigmoid)
        self.trainClassifier(sigmoid, net)
	for j in range(len(net.hiddenLayer)):
            if net.hiddenLayer[j].getClass() == net.currentClass:
    	        net.hiddenLayer[j].delta = net.hiddenLayer[j].delta * .8 + (death *.2 * gradient) 

    
    def trainClassifier(self,sigmoid, net):
	currentClass = net.currentClass
	perceptrons = net.perceptrons
	if sigmoid > .9:
	    return
	gradient = Util.sigmoid_prime(sigmoid)
	perceptrons[currentClass].delta = perceptrons[currentClass].delta *.8 + (net.lastResult * .2 * gradient)

    def gradientDelta(self,net):
	c1 = net.currentClass
	orgDeaths = net.orgDeathOccurances
	gradient = float(orgDeaths - self.deathOccurances) /float(orgDeaths)
	gradient_s = '%.4f' % gradient
	print("Training Class: %s, old Deaths: %d, new Deaths: %d, gradient: %s)"%(net.classLabels[c1], orgDeaths, self.deathOccurances, gradient_s))
	
	net.perceptrons[c1].delta = net.perceptrons[c1].delta * gradient
	for j in range(len(net.hiddenLayer)):
            if net.hiddenLayer[j].getClass() == c1:
    	        net.hiddenLayer[j].delta = net.hiddenLayer[j].delta * gradient
    
    def execute(self, net):
	if self.done:
	    return net.GradientState
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
	
