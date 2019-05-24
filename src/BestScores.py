import numpy as np
import os

class BestScores(object):

    def __init__(self, exercise):
	self.path = exercise + "/best.npy"
	if os.path.isfile(self.path):
	    best = np.load(self.path).tolist()
	    self.best = best
	else:
	    self.best = []
	    self.best.append(-500000)
	for b in self.best:
	    print("Best %d" %b)
        return None

    def getBest(self):
	return np.max(self.best)

    def setBest(self, score):
	self.best.append(score)

    def save(self):
	np.save(self.path, np.array(self.best))


