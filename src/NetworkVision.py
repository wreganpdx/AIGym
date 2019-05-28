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

class NetworkVision(object):
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
    global deaths 
    deaths =  []
    global scoresThisRound
    scoresThisRound = []
    global roundMax
    roundMax = 25
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
    global maxPopulation
    maxPopulation = 10
    global path
    path = ""
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
        space = np.random.rand()
        discount = int(45 * space) + 15
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
	global deaths 
	deaths = []
	global scoresThisRound
	scoresThisRound = []
	global roundMax
	roundMax = 25
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
	global maxPopulation
	maxPopulation = 16
	global path
	path = ""
	_nodes = _nodes * _metaLayers
        for k in range(len(classes)):
            p = Neuron(classes[k], classes[k], _nodes)
            perceptrons.append(p)

        for i in range(_nodes):
            h = HiddenNetwork(i * 1000 + i, i %len(classes), weights)
            hiddenLayer.append(h)
	classCodes = NetworkVision.defineBrand()

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
    def matchPair():
	global classCodes
	global alreadyMutated
	global currentMutation
	np.random.shuffle(classCodes)
	for i in range(len(classCodes)):
	    np.random.shuffle(classCodes[i])
	for i in range(len(classCodes)):
	    for k in range(len(classCodes[i])):
	    	str1 = "%d"%classCodes[i][k]
		for x in range(len(classCodes)):
		    if x == i:
			continue
		    for y in range(len(classCodes[x])):
			
			str2 = "%d"%classCodes[x][y]
			pair = str1 + "_" + str2

			if pair in alreadyMutated:
			    continue

			pair = str2 + "_" + str1
			if pair in alreadyMutated:
			    continue

			if NetworkVision.deathsEqual(str1, str2):
			    continue
			
			alreadyMutated[pair] = 0
			currentMutation = pair
			return (str1, str2)

    @staticmethod
    def deathsEqual(str1, str2):
	int1 = int(str1)
	int2 = int(str2)
	d1 = int(int1 / 10)
	d2 = int(int2 / 10)
	return d1 == d2

    @staticmethod
    def trainPair(str1, str2):
	NetworkVision.trainOne(str1)
	NetworkVision.trainOne(str2)

    @staticmethod
    def trainOne(str1):
	global classes
	global deaths
	int1 = int(str1)
	c1 = int1 % 10
	d1 = ((int1 % 100) - c1) /10
	assert d1 < len(classes)
        if d1 >= len(deaths):
	    print(d1)
	    print(int1)
	    print(len(deaths))
	    assert False
	p1 = int(int1 /100)
	if p1 == 0:
	    p1 = -1


        perceptrons[c1].anealDelta(1, p1)
	for j in range(len(hiddenLayer)):
            if hiddenLayer[j].getClass() == c1:
    	        hiddenLayer[j].delta = (deaths[d1] * p1)  
	
		
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
	if bestScore < float(scoreObj.getBest()):
	    print("scores too low, not saving")
	    return
	p = []
	h = []
	for i in range(len(perceptrons)):
	    p.append(perceptrons[i].w)
	for k in range(len(hiddenLayer)):
	    h.append(hiddenLayer[k].w)
	avg_s = '%.4f'% bestScore
	np.save(path+avg_s + "perc", p)
	np.save(path+avg_s + "hid", h)
	scoreObj.setBest(avg_s)
	scoreObj.save()
	

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
    def loadBestMutationDelta():
	global path
	global perceptrons
	global hiddenLayer
	print("checking for weights %s" % path)
	if os.path.isfile(path + "dt_perc.npy") != True:
	    return
	print("loading weights")
	p = np.load(path + "dt_perc.npy")
	h = np.load(path + "dt_hid.npy")
	for i in range(len(perceptrons)):
	    perceptrons[i].delta = p[i]
	a1 = len(hiddenLayer)
	a2 = len(h)
	r = min(a1, a2)
	for k in range(r):
	    hiddenLayer[k].delta = h[k]
	NetworkVision.saveBest()

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
    	discount = int(45 * space) + 15
	avgObs = avgObs * 0


    @staticmethod
    def applyAnnealing(T, steps, rew, scores):
	global state
	global GradientState
	global TestState
	global currentMutation
	global alreadyMutated
	global maxPopulation
	global scoresThisRound
	finishedTest = False
	finishedGradient = False
	if state == TestState:
	    finishedTest = NetworkVision.testState(T,steps,rew,scores)
	    #print("Test: %d" % len(scoresThisRound))
	else:	
	    NetworkVision.zeroDelta()
	    finishedGradient = NetworkVision.gradientState(T,steps,rew,scores)
	
	if finishedTest:
	    if len(alreadyMutated) < maxPopulation:
		NetworkVision.zeroDelta()
	    	mut1, mut2 = NetworkVision.matchPair()
	    	NetworkVision.trainPair(mut1,mut2)
	    else:
		NetworkVision.completeTraining()
		NetworkVision.resetVision()
	    scoresThisRound = []
	if finishedGradient:
	    NetworkVision.clusterdeaths()
	    NetworkVision.zeroDelta()
	    mut1, mut2 = NetworkVision.matchPair()
	    NetworkVision.trainPair(mut1,mut2)
		
	return

    @staticmethod
    def resetVision():
	global deaths
	global Frames
	global alreadyMutated
	global deathFrames
	global bestScore
	alreadyMutated = {}
	Frames = []
	deaths = []
	deathFrames = []
	bestScore = 0
	bestMutation = ""
	currentMutation = ""
	state = TestState

    @staticmethod
    def completeTraining():
	global state
	global GradientState
	NetworkVision.loadBestMutationDelta()
	state = GradientState

    @staticmethod
    def testState(T, steps, rew, scores):
	global scoresThisRound
	global roundMax	
	global bestScore
	global bestMutation
	global currentMutation
	global state
	global gradientState	
	scoresThisRound.append(rew)
	if len(scoresThisRound) >= roundMax:
	    avg = 0
	    for i in range(len(scoresThisRound)):
                avg += scoresThisRound[i]
	    avg = float(avg) / float(len(scoresThisRound))
	    if avg > bestScore:
		bestScore = avg    
		bestMutation = currentMutation	
		NetworkVision.saveBestMutationDelta()
	    avg_s = '%.4f' % avg
	    print("CurrentMutation: %s, new avg score: %s" % (currentMutation, avg_s))
	    return True
	return False

    @staticmethod
    def gradientState(T, steps, rew, scores):
	global deathFrames
	global TestState
	global state
        NetworkVision.createDeathFrame()
	NetworkVision.setGradient()
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
	global deaths
	_deaths = []
	for i in range(len(deathFrames)):
	    _deaths.append(deathFrames[i].flatten())
	shape = deathFrames[0].shape
	kmeans = KMeans(n_clusters=len(classes)).fit(_deaths)
	centroids = []
	for i in range(len(kmeans.cluster_centers_)):
	    centroids.append(np.reshape(kmeans.cluster_centers_[i], shape))
	print("Death Clusters Created")
	#for i in range(len(centroids)):
	#    plt.imshow(centroids[i])
	 #   plt.figure()
	  #  plt.show()
	deaths = np.array(centroids, copy=True)

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
	NetworkVision.saveWeightsToDisk()


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
	unique = np.unique(obs)
	originalDelta = obs - lastObs
	deltaObs = originalDelta[:]
	if state == GradientState:
	    deltaObs = np.abs(deltaObs)
	    Frames.append(avgObs[:])
            if len(Frames) >= discount:
		Frames.pop(0)
		    
	    avgObs = avgObs * .8 + deltaObs * .2 
		
	    
        result = np.empty(len(hiddenLayer))
	myRange = len(hiddenLayer) 
	
	d = originalDelta.flatten()
	if allowRandom and not np.any(d):
 	    return rand
	d = d.transpose()
        for i in range(myRange):
            result[i] = hiddenLayer[i].compute(d)

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

