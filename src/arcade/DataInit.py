import sys
import numpy as np
# count the arguments
class DataInit(object):

    def __init__(self):
        return None

    @staticmethod
    # simple class to quantify data based on command line input for data.
    def getData():
        arguments = len(sys.argv) - 1
        if arguments < 2:
            print("Not enough arguments, please include <Exercise> and <Number of Episodes>")
            exit(1)


        exercise = sys.argv[1]
        numIterations = sys.argv[2]
	render = True
	printStuff = True
	obsDefault = "deltaObs"
	if arguments > 2 and int(sys.argv[3]) != 1:
            render = False

	if arguments > 3 and int(sys.argv[4]) != 1:
            printStuff = False
	
	if arguments > 4:
	    if sys.argv[4] == "deltaObs":
	         obsDefault = sys.argv[4]
	    if sys.argv[4] == "obs":
	         obsDefault = sys.argv[4]
	    if sys.argv[4] == "diffObs": 
	         obsDefault = sys.argv[4]


        return (exercise, numIterations, render, printStuff, obsDefault)
