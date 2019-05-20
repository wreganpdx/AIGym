import numpy as np
import time
class Util(object):

    @staticmethod
    def clamp(num, min_value, max_value):
        return max(min(num, max_value),
                   min_value)  # basic clamp function for doing error adjustment (not used in asignment, but useful for future)
    @staticmethod
    def shuffle_in_unison(x, y):
        np.random.seed(int(time.time()))
        state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(state)
        np.random.shuffle(y)

    @staticmethod
    def mag(self):
        return self.prediction * self.prediction

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Util.sigmoid(z) * (1 - Util.sigmoid(z))

    @staticmethod
    def clamp( num, min_value, max_value):
        return max(min(num, max_value), min_value) #basic clamp function for doing error adjustment (not used in asignment, but useful for future)



    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
