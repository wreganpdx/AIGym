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

class NetworkDvt(object):
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
    global Frames
    Frames = []
    global deathFrames
    deathFrames = []
    global mutations
    mutations = []
    global deathRange
    deathRange = 32
    global alreadyMutated
    alreadyMutated = {}
    global death 
    death =  []
    global scoresThisRound
    scoresThisRound = []
    global roundMax
    roundMax = 50
    global bestScore
    bestScore = ""
    global bestMutation
    bestMutation = ""
    global currentMutation
    currentMutation = ""
    global state
    state = 0
    global GradientState
    GradientState = 0
    global TestState
    TestState = 1
    global path
    path = ""
    global bestAvg
    bestAvg = 0
    global bestDeltas
    bestDeltas = []
    global deltaGradients
    deltaGradients = []
    global currentClass
    currentClass = 0
    global currentDeath 
    currentDeath = 0
    global deathOccurances
    deathOcccurances = 0
    global orgDeathOccurances
    orgDeathOccurances = 0 
    global discountSpace
    discountSpace = 45
    global kmeans
    global classLabels
    global avgDiscount
    global sigmoids 
    global lastResult
    global deathDelta
    sigmoids = []
    def __init__(self, _nodes, _classes, weights, obs, _metaLayers, trace, _Random):
	global discountSpace
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
        space = np.random.rand()
        discount = int(discountSpace * space) + 15
	global gradientCount
	gradientCount = 0
	global Frames
	Frames = []
	global deathFrames
	deathFrames = []
	global mutations
	mutations = []
	global deathRange
	deathRange = 32
	global classCodes
	classCodes = []
        global alreadyMutated
        alreadyMutated = {}
	global death 
	death = []
	global scoresThisRound
	scoresThisRound = []
	global roundMax
	roundMax = 50
	global bestScore
	bestScore = 0
        global bestMutation
	bestMutation = ""
	global currentMutation
	currentMutation = ""
	global state
	state = 0
	global GradientState
	GradientState = 0
	global TestState
	TestState = 1
	global path
	path = ""
    	global bestAvg
    	bestAvg = 0
	global bestDeltas
	bestDeltas = []
	global deltaGradients
	deltaGradients = []
    	global currentClass
    	currentClass = 0
	global currentDeath
	currentDeath = 0
        global deathOccurances
        deathOccurances = 0
   	global orgDeathOccurances
	global avgDiscount
	global lastResult
	global deathDelta
	deathDelta = np.empty(weights)
	lastResult = np.empty(weights)
	avgDiscount = (discount + discountSpace)/2
   	orgDeathOccurances = 0 
        for k in range(len(classes)):
            p = Neuron(classes[k], classes[k], _nodes)
            perceptrons.append(p)

        for i in range(_nodes):
            h = HiddenNetwork(i * 1000 + i, i %len(classes), weights)
            hiddenLayer.append(h)
	classCodes = NetworkDvt.defineBrand()
	

    @staticmethod
    def defineBrand():
	global classes
	global deathRange
	codes = []
	for i in range(len(classes)):
	    code = i
	    class_codes = []
	    for j in range(len(classes)):
                code += j*10
		for k in range (2):
		    code += k * 100
		    class_codes.append(code)
		    code -= k * 100
		code -= j * 10
	    codes.append(class_codes)
	print(codes)
	return codes

    @staticmethod  
    def getCurrent():
	global classCodes
	global alreadyMutated
	global currentMutation
	global currentClass	
	global currentDeath
        ret = currentClass
	ret += 10 * currentDeath
	rand = np.random.rand()
	if rand > .5:
	    ret += 100
	currentMutation = "%d" % ret
	return currentMutation


    @staticmethod
    def trainOne(c1):
	global classes
	global death
	for j in range(len(hiddenLayer)):
            if hiddenLayer[j].getClass() == c1:
    	        hiddenLayer[j].delta = death  
    


    @staticmethod
    def continueTraining(sigmoid):
	global classes
	global death
	global sigmoids
	global currentClass
	if sigmoid > .9:
	   return

	gradient = Util.sigmoid_prime(sigmoid)
        NetworkDvt.trainClassifier(sigmoid)
	for j in range(len(hiddenLayer)):
            if hiddenLayer[j].getClass() == currentClass:
    	        hiddenLayer[j].delta = hiddenLayer[j].delta * .8 + (deathDelta *.2 * gradient) 

    @staticmethod
    def trainClassifier(sigmoid):
	global currentClass
	global deathDelta
	global perceptrons
	global roundMax
	global originalDeathOccurances
	if sigmoid > .9:
	    return
	gradient = Util.sigmoid_prime(sigmoid)
	perceptrons[currentClass].delta = perceptrons[currentClass].delta *.8 + (lastResult * .2 * gradient)
    
    @staticmethod
    def gradientDelta():
	global currentMutation
	global classLabels
	global currentClass
	
	global deathOccurances
	global orgDeathOccurances
	c1 = currentClass
	gradient = float(orgDeathOccurances - deathOccurances) /float(orgDeathOccurances)
	gradient_s = '%.4f' % gradient
	print("Training Class: %s, old Deaths: %d, new Deaths: %d, gradient: %s)"%(classLabels[currentClass], orgDeathOccurances, deathOccurances, gradient_s))
	
	perceptrons[c1].delta = perceptrons[c1].delta * gradient
	for j in range(len(hiddenLayer)):
            if hiddenLayer[j].getClass() == c1:
    	        hiddenLayer[j].delta = hiddenLayer[j].delta * gradient
	
    @staticmethod 
    def setPath(_path):
    	global path
	path = _path

    @staticmethod 
    def setScoreObj(obj):
    	global scoreObj
	scoreObj = obj

    @staticmethod 
    def saveWeightsToDisk():
	global path
	global perceptrons
	global hiddenLayer
	global bestScore
	global scoreObj
	#if bestScore < float(scoreObj.getBest()):
	 #   print("scores too low, not saving")
	  #  return
	p = []
	h = []
	for i in range(len(perceptrons)):
	    perceptrons[i].w 
	    p.append(perceptrons[i].w)
	for k in range(len(hiddenLayer)):
	    hiddenLayer[k].w 
	    h.append(hiddenLayer[k].w)
	avg_s = '%.4f'% bestScore
	np.save(path+avg_s + "perc", p)
	np.save(path+avg_s + "hid", h)
	scoreObj.setBest(avg_s)
	scoreObj.save()
	print("Saving Weights to disk " + path + avg_s)
	

    @staticmethod 
    def saveBestMutationDelta():
	global path
	global perceptrons
	global hiddenLayer
	p = []
	h = []
	for i in range(len(perceptrons)):
	    p.append(perceptrons[i].delta)
	for k in range(len(hiddenLayer)):
	    h.append(hiddenLayer[k].delta)
	np.save(path + "dt_perc", p)
	np.save(path + "dt_hid", h)

    @staticmethod 
    def loadBestMutationDelta(omit):
	global path
	global perceptrons
	global hiddenLayer
	print("checking Mutation weights: %s omiting: %d" % (path, omit))
	if os.path.isfile(path + "dt_perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(path + "dt_perc.npy")
	h = np.load(path + "dt_hid.npy")
	for i in range(len(perceptrons)):
	    if i == omit:
		continue;
	    perceptrons[i].delta = p[i]
	a1 = len(hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    if k == omit:
		continue;
	    hiddenLayer[k].delta = h[k]

    @staticmethod 
    def loadWeightsFromDisk(_path):
	global path
	global perceptrons
	global hiddenLayer
	path = _path
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
    def setGradient():
	global discount
	global gradientCount
	global avgObs
	global Frames
    	space = np.random.rand()
    	discount = int(discountSpace * space) + 15
	avgObs = avgObs * 0


    @staticmethod
    def applyAnnealing(T, steps, rew, scores):
	global state
	global GradientState
	global TestState
	global currentMutation
	global alreadyMutated
	global scoresThisRound
	global currentClass
	finishedTest = False
	finishedGradient = False
	if state == TestState:
	    finishedTest = NetworkDvt.testState(T,steps,rew,scores)
	    #print("Test: %d" % len(scoresThisRound))
	else:	
	    finishedGradient = NetworkDvt.gradientState(T,steps,rew,scores)
	
	if finishedTest:
            NetworkDvt.completeTraining()
	    NetworkDvt.trainOne(currentClass)
	    scoresThisRound = []
	if finishedGradient:
	    NetworkDvt.clusterdeaths()
	    NetworkDvt.saveBestMutationDelta()
	    NetworkDvt.trainOne(0)
		
	return
    @staticmethod
    def resetForClass():
	global Frames
	global alreadyMutated
	global deathFrames
	global bestScore
	global bestAvg
	global bestMutation
	global currentMutation
	global deathOccurances
	Frames = []
	deathFrames = []
	bestScore = 0
	bestAvg = 0
	bestMutation = ""
	currentMutation = ""
	deathOccurances = 0

    @staticmethod
    def resetVision():

	global alreadyMutated
	alreadyMutated = {}
	global Frames
	Frames = []
	global deathFrames
	deathFrames = []
	global bestScore
	bestScore = 0
	global bestAvg
	bestAvg = 0
	global bestMutation
	bestMutation = ""
	global currentMutation
	currentMutation = ""
	global currentClass
	currentClass = 0
	global deathOccurances
	deathOccurances = 0

    @staticmethod
    def completeTraining():
	global state
	global GradientState
	global currentClass
	global classes
	NetworkDvt.gradientDelta()
	if currentClass == len(classes)-1:
	    print("currentClass: %d" % currentClass)
	    NetworkDvt.saveBest()
	    state = GradientState
	    NetworkDvt.resetVision()
	    NetworkDvt.zeroDelta()

	else:
	    print("currentClass: %d" % currentClass)
	    currentClass += 1
	    NetworkDvt.resetForClass()

    @staticmethod
    def testState(T, steps, rew, scores):
	global scoresThisRound
	global roundMax	
	global bestScore
	global bestMutation
	global currentMutation
	global bestAvg
	global currentDeath
	global kmeans
	global deathFrames
	global deathOccurances
	global sigmoids
	NetworkDvt.createDeathFrame()
	scoresThisRound.append(rew)
	_death = []
	_death.append(deathFrames[len(deathFrames)-1].flatten())
	d = kmeans.predict(_death)
	d = d[0]
	if d == currentDeath:
	    deathOccurances += 1
	    NetworkDvt.continueTraining(sigmoids[len(sigmoids)-1])
	else:
	    sigmoids.pop()
	if len(scoresThisRound) >= roundMax:

            success = float(np.count_nonzero(scoresThisRound))/float(len(scoresThisRound))
	    avg = np.average(scoresThisRound)
	    print(scoresThisRound)
	    if success > bestScore or (success == bestScore and bestAvg < avg):
		bestScore = success
		bestAvg = avg    
		bestMutation = currentMutation	
	    avg_s = '%.4f' % avg
            success_s = '%.4f' % success
	    print("Current Mutation: %s, avg: %s, reward: %s" % (currentMutation, avg_s,success_s))
	    return True
	
	return False

    @staticmethod
    def gradientState(T, steps, rew, scores):
	global deathFrames
	global TestState
	global state
        NetworkDvt.createDeathFrame()
	NetworkDvt.setGradient()
	if len(deathFrames) >= roundMax:
	   state = TestState
	   return True
	#else:
	   #print("Death Frames: %d" %len(deathFrames))
	   

	return 

    @staticmethod
    def clusterdeaths():
	global deathFrames
	global classes
	global death
	global kmeans
	global orgDeathOccurances
	global currentDeath
	_deaths = []
	frequencies = np.zeros(len(classes))
	for i in range(len(deathFrames)):
	    _deaths.append(deathFrames[i].flatten())
	shape = deathFrames[0].shape
	deathFrames = []
	kmeans = KMeans(n_clusters=len(classes)).fit(_deaths)
  
	predictions = kmeans.predict(_deaths)
	for i in range(len(predictions)):
	    frequencies[predictions[i]] = frequencies[predictions[i]] + 1
    #frequencies[index] = frequencies[index] + 1
	centroids = []
	for i in range(len(kmeans.cluster_centers_)):
	    centroids.append(np.reshape(kmeans.cluster_centers_[i], shape))
	print("Death Clusters Created")
	
	orgDeathOccurances = np.max(frequencies)
	currentDeath = np.argmax(frequencies)
	#for i in range(len(centroids)):
	 #   plt.imshow(centroids[i])
	  #  plt.figure()
	   # plt.show()
	death = np.array(centroids[currentDeath], copy=True)

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
	NetworkDvt.saveWeightsToDisk()


    @staticmethod
    def createDeathFrame():
	global Frames
	global deathFrames
	global avgObs
	length = len(Frames)
	if length == 0:
	    length = 1
	avgObs = avgObs * 0
	for i in range(len(Frames)):
	    avgObs = avgObs + Frames[i]
	avgObs = avgObs/length
	deathFrames.append(avgObs[:])	


    @staticmethod
    def calculateObs(obs, rand = 0):
        global hiddenLayer
        global perceptrons
        global classes
	global lastObs
	global avgObs
	global discount
	global gradientCount
	global Frames
	global state
	global GradientState
	global TestState
	global sigmoids
	global avgDiscount
	global lastResult
	unique = np.unique(obs)
	originalDelta = obs - lastObs
	deltaObs = originalDelta[:]
    	deltaObs = np.abs(deltaObs)
    	Frames.append(avgObs[:])
    	if len(Frames) >= discount:
	    Frames.pop(0)
	
	    
    	avgObs = avgObs * .8 + deltaObs * .2 
		
	    
        lastResult = np.empty(len(hiddenLayer))
	myRange = len(hiddenLayer) 
	
	d = originalDelta.flatten()
	if allowRandom and not np.any(d):
 	    return rand
	d = d.transpose()
        for i in range(myRange):
            lastResult[i] = hiddenLayer[i].compute(d)

	lastObs = obs[:]

        scores = np.empty(len(classes))
        bestScore = perceptrons[0].compute(lastResult)
        scores[0] = bestScore
        bestIndex = 0
	if currentClass == 0:
	    sigmoids.append(bestScore)
        for i in range(1, len(classes)):
            score = perceptrons[i].compute(lastResult)
	    if currentClass == i:
		sigmoids.append(score)
            if score > bestScore:
                bestScore = score
                bestIndex = i
	if len(sigmoids) > avgDiscount:
	    sigmoids.pop(0)
        perceptrons[bestIndex].fitness += 1

        return bestIndex


    @staticmethod
    def getPerceptrons():
        return perceptrons



    @staticmethod
    def numNeurons():
        return len(perceptrons)


    @staticmethod
    def setClassLabels(labels):
	global classLabels
	classLabels = labels

