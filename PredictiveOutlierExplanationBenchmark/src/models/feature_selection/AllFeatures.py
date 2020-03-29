import numpy as np


class AllFeatures:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        return np.array(range(0, X_train.shape[1])), None
