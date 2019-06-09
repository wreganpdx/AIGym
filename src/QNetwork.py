"""
William C Regan
Student ID: 954681718
Artificial Intelligence
Portland State University
Final Project
"""
import numpy as np
from q_learn import q_learn

class QNetwork(object):
    global classes
    global Frames
    global QLearn
    global lastAction
    global weights
    global clusters
    global actions
    global rewards
    global states
    global bestScore
    global lastFrame
    global obsStack
    global stackSize
    global useDiff
    global elitism
    global distances
    global epsilon


    def __init__(self, _clusters, _classes, _weights, stacks, diff, _test, _discrete, _elitism, _epsilon):
        global classes
        classes = np.arange(_classes)
        global Frames
        Frames = []
        global classes
        classes = _classes
        global weights
        weights = _weights
        global clusters
        clusters = _clusters
        global QLearn
        QLearn = q_learn(classes, clusters)
        global lastAction
        global actions
        actions = []
        global rewards
        rewards = []
        global states
        states = []
        global bestScore
        global originalBestScore
        originalBestScore = -500000
        bestScore = originalBestScore
        global bestWeights
        global bestQvalues
        global lastReward
        global testingWeights
        testingWeights = False
        global testWeights
        global testScores
        global testValues
        global obsStack
        global stackSize
        global useDiff
        global lastFrame
        global test
        global discrete
        global elitism
        global distances
        global epsilon
        epsilon = _epsilon
        distances = []
        elitism = _elitism
        discrete = _discrete
        test = _test
        lastFrame = np.zeros(weights)
        useDiff = diff
        stackSize = stacks
        obsStack = []
        for i in range(stacks):
            obsStack.append(np.zeros(weights))
        testScores = []
        lastReward = originalBestScore
        actions.append(0)
        weights = np.zeros(weights)
        weights = len(weights)
        print(stackSize)
        f = np.zeros((stackSize, weights), dtype='f')
        print(f)
        Frames.append(f.flatten())
        states.append(0)

    @staticmethod
    def applyAnnealing(T, steps, rew, totalReward):
        global Frames
        global actions
        global rewards
        global weights
        global bestScore
        global bestQvalues
        global originalBestScore
        global bestWeights
        global lastReward
        global testingWeights
        global testScores
        global testValues
        global testWeights
        global stackSize
        global obsStack
        global test
        global elitism
        global distances
        rewards.append(rew);
        reset = False
        avgDist = np.average(distances)
        if testingWeights:
            testScores.append(totalReward)
            if len(testScores) > 10:
                check = np.median(testScores)
                if test == "Mean":
                    check = np.average(testScores)
                if test == "Min":
                    check = np.min(testScores)
                if test == "Max":
                    check = np.max(testScores)
                if check > bestScore:
                    bestScore = check
                    bestWeights = testWeights
                    bestQvalues = testValues[:]
                    reset = True
                    testingWeights = False
                    avg_s = '%.4f'%check
                    print("saving weights: Score: %s" % avg_s)
                    testScores = []
                else:
                    testingWeights = False
                    testScores = []
                    reset = True
                    q_learn.setWeights(bestWeights, bestQvalues[:])
            else:
                reset = True
                q_learn.setWeights(testWeights, testValues[:])
        else:
            if totalReward > bestScore:
               try:
                    testWeights,testValues = q_learn.getWeights()
                    if elitism :
                        testingWeights = True
                    print("t, %d"% totalReward),
               except:
                   print("weights aren't ready, can't save.")
            else:
                if lastReward > totalReward:
                    anealing = float(T)/float(steps)
                    rand = np.random.random()
                    if rand < anealing:
                        if bestScore !=originalBestScore:
                            reset = True
                            print("r"),
                            q_learn.setWeights(bestWeights, bestQvalues[:])

        rand = np.random.rand()
        anealing = 1 - (float(T) / float(steps))
        if rand < anealing and lastReward >= totalReward:
            if reset != True:
                QLearn.kmeansInstance(Frames, rewards, actions)
        lastReward = totalReward
        Frames = []
        actions = []
        rewards = []
        obsStack = []
        distances = []
        for i in range(stackSize):
            obsStack.append(np.zeros(weights))
        Frames.append(np.zeros((stackSize, weights), dtype='f'))
        actions.append(0)
        states.append(0)


    @staticmethod
    def calculateObs(obs, rew, T, duration):
        global classes
        global QLearn
        global lastAction
        global weights
        global actions
        global rewards
        global states
        global weights
        global obsStack
        global stackSize
        global useDiff
        global testingWeights
        global discrete
        global distances
        global epsilon
        if useDiff:
            diffFrame = obs - obsStack[len(obsStack)-1]
        lastFrame = obs[:]

        obsStack.append(lastFrame[:])
        if len(obsStack) > stackSize:
            obsStack.pop(0)
           # print (obsStack)
        newFrame = obsStack[:]
        if useDiff:
            newFrame[0] = diffFrame


        newFrame = np.asarray(newFrame).flatten()
        lastAction = 0
        if len(actions) > 0:
            lastAction = actions[len(actions) - 1]
        else:
            lastAction = 0
        if len(states) > 0:
            lastState = states[len(states)-1]
        else:
            lastState = 0
        state, dist = QLearn.classifyWithKMeans(newFrame)
        distances.append(dist)
        if (discrete == True and testingWeights) != True:
            q_learn.calcLastReward(lastAction, lastState, state, rew) #(lastAction, lastObs, nextObs, rew):

        eps = epsilon
        aneal = 1 - (float(T)/float(duration))
        eps = eps * aneal
        lastAction = QLearn.pickNewAction(state, eps)
        actions.append(lastAction)
        rewards.append(rew)
        states.append(state)
        Frames.append(newFrame)

        return lastAction


