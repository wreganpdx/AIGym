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


        return (exercise, numIterations)
