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
	    t = str(b)
	    print("Best %s" %t)
        return None

    def getBest(self):
	b = []
	for s in self.best:
	    b.append(float(s))
	return np.max(b)

    def setBest(self, score):
	self.best.append(score)

    def save(self):
	np.save(self.path, np.array(self.best))


