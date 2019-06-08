"""
William C Regan
Student ID: 954681718
Artificial Intelligence
Portland State University
Final Project
"""

import numpy as np
import random
from scipy.spatial import distance

from sklearn.cluster import KMeans

class q_learn(object):
    global epsilon
    global clusters
    global actions
    global lastCluster
    global cluster_action_values
    global kmeans
    global createdKmeans
    global newKmeans
    global oldKmeans
    def __init__(self, _actions, _clusters):
        global clusters
        global actions
        global lastAction
        global lastCluster
        global cluster_action_values
        global kmeans
        global newKmeans
        global createdKmeans
        global oldKmeans
        createdKmeans = False
        clusters = _clusters
        actions = _actions
        cluster_action_values = np.zeros((_clusters, _actions))

        return

    @staticmethod
    def getWeights():
        global kmeans
        global cluster_action_values
        return kmeans, cluster_action_values[:]

    @staticmethod
    def setWeights(_w, _v):
        global kmeans
        global cluster_action_values
        kmeans = _w
        cluster_action_values = _v

    @staticmethod
    def kmeansInstance(states, _rewards, _actions):
        global createdKmeans
        global clusters
        if createdKmeans == True:
            q_learn.meldNewClusters(states)
        else:
            #only called once ever
            q_learn.createNewClusters(_rewards, states, _actions)

    @staticmethod
    def createNewClusters(_rewards, states, _actions):
        global newKmeans
        global clusters
        global cluster_action_values
        global actions
        global createdKmeans
        assert createdKmeans != True
        z = np.asarray(states[0]).flatten()
        n = np.empty((len(states), len(z)))
        for i in range(len(states)):
            n[i] = states[i].flatten()
        newKmeans = KMeans(n_clusters=clusters).fit(n)
        stuff = newKmeans.predict(n)
        clusterNum = np.zeros((clusters, actions))
        for i in range(len(stuff)):

            cluster_action_values[stuff[i]][_actions[i]] += _rewards[i]
            clusterNum[stuff[i]][_actions[i]] += 1
        for i in range(clusters):
            for j in range(actions):
                if clusterNum[i][j] != 0:
                    cluster_action_values[i][j] = cluster_action_values[i][j] / clusterNum[i][j]
        createdKmeans = True
        return


    @staticmethod
    def meldNewClusters(states):
        global kmeans
        global clusters
        global actions
        global newKmeans
        global cluster_action_values
        global oldKmeans
        oldKmeans = newKmeans
        z = np.asarray(states[0]).flatten()
        n = np.empty((len(states), len(z)))
        for i in range(len(states)):
            n[i] = states[i].flatten()
        newKmeans = KMeans(n_clusters=clusters).fit(n)
        #q_learn.createNewClusters(states)
        old_action_values = cluster_action_values[:]
        cluster_action_values = np.zeros((clusters, actions))
        clusterVals = np.zeros((clusters, actions))

        clusterNum = np.zeros(clusters)
        for i in range(len(oldKmeans.cluster_centers_)):
            t = []
            t.append(oldKmeans.cluster_centers_[i])
            cls = newKmeans.predict(t)[0]
            for j in range(actions):
                clusterVals[cls][j] = clusterVals[cls][j] + old_action_values[i][j]
            clusterNum[cls] += 1
        for i in range(clusters):
            if clusterNum[i] != 0:
                clusterVals[i] = clusterVals[i] / clusterNum[i]
                for j in range(actions):
                    cluster_action_values[i][j] = clusterVals[i][j]
        kmeans = newKmeans
        return

    @staticmethod
    def pickNewAction(current, epsilon):
        biggest = np.max(cluster_action_values[current])
        if np.any(cluster_action_values[current]):
            optimal = int(np.random.rand() * len(cluster_action_values[current]))
            while cluster_action_values[current][optimal] != biggest:
                optimal = int(np.random.rand() * len(cluster_action_values[current]))
            if optimal < 0:
                print(optimal)
        else:
            optimal = np.random.randint(0, high=len(cluster_action_values[current]))

        rand = np.random.rand()

        if rand < epsilon:
            optimal = np.random.randint(0, high=len(cluster_action_values[current]))

        return optimal


    @staticmethod
    def classifyWithKMeans(feature):
        global actions
        try:
            f = []
            f.append(feature)
            index = kmeans.predict(f)[0]
            cluster = kmeans.cluster_centers_[index]
            dst = distance.euclidean(feature, cluster)
            return kmeans.predict(f)[0], dst
        except:
            return int(np.random.rand() * clusters), 0


    @staticmethod
    def calcLastReward(lastAction, lastObs, nextObs, rew):
        global cluster_action_values
        old = cluster_action_values[nextObs][lastAction]
        cluster_action_values[lastObs][lastAction]= (old * .8) + .2 * (rew + .9 * np.max(cluster_action_values[nextObs]))
