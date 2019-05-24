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
	if arguments > 2 and int(sys.argv[3]) != 1:
            render = False

	if arguments > 3 and int(sys.argv[3]) != 1:
            printStuff = False


        return (exercise, numIterations, render, printStuff)
