import sys
import numpy as np
# count the arguments
class DataInit(object):

    def __init__(self):
        return None

    @staticmethod
    # simple class to quantify data based on command line input for data.
    def getData():
        exercise = "Acrobot-v1"
        numIterations = 2
        render = False
        clusters = 25
        stacks = 2
        diff = True
        ready = False
        test = "Median"
        discrete = True
        elitism = True
        epsilon = .1
        eps_s = '%.4f' % epsilon
        while ready != True:
            print("Current Settings: ")
            print("(M)inutes: %d" % numIterations)
            print("(E)xercise: %s"% exercise)
            print("(R)ender: %r"%render)
            print("(C)lusters: %d"% clusters)
            print("(F)rame Depth: %d"%stacks)
            print("(D)ifference Function: %r" % diff)
            print("(T)est Verification: %s" % test)
            print("D(i)screte testing: %r" % discrete)
            print("E(L)itism: %r" % elitism)
            print("(Z)ero Epsilon: %s"% eps_s)
            print("(S)tart")
            print("Enter a command: E, R, C, F, S, T, I, M, L, Z")
            g = raw_input(":")
            if g == "Z" or g == "z":
                print("What epsilon value should the simulation use?")
                m = raw_input(":")
                epsilon = float(m)
                eps_s = '%.4f' % epsilon
            if g == "m" or g == "M":
                print("How many minutes should the simulation run?")
                m = raw_input(":")
                numIterations = int(m)
            if g == "L" or g == "l":
                if elitism:
                    elitism = False
                else:
                    elitism = True
            if g == "E" or g == "e":
                print("Enter an exercise")
                e = raw_input(":")
                exercise = e
            if g == "T" or g == "t":
                print("Would you like to use (A)verage, (L)owest, (M)edian or MA(X) for confirmation?")
                m = raw_input(":")
                if m == "A" or m == "a":
                    test = "Mean"
                if m == "L" or m == "l":
                    test = "Min"
                if m == "M" or m == "m":
                    test = "Median"
                if m == "X" or m == "x":
                    test = "Max"
            if g == "R" or g == "r":
                if render:
                    render = False
                else:
                    render = True
            if g == "C" or g == "c":
                print("Enter a number of custers between 1-100 (suggested)")
                c = raw_input(":")
                clusters = int(c)
            if g == "F" or g == "f":
                print("Enter a frame depth 1-4 suggested (2 minimum to use difference function)")
                f = raw_input(":")
                stacks = int(f)
            if g == "D" or g == "d":
                if diff:
                    diff = False
                else:
                    diff = True
            if g == "I" or g == "i":
                if discrete:
                    discrete = False
                else:
                    discrete = True
            if g == "S" or g == "s":
                ready = True


        return (exercise, numIterations, render, stacks, diff, clusters, test, discrete, elitism, epsilon)
