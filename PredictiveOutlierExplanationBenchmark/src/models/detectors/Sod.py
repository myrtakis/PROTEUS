from pyod.models.sod import SOD
import numpy as np


class Sod:

    def __init__(self):
        self.__clf = None

    def train(self, X_train, params):
        self.__clf = SOD(alpha=params['alpha'], n_neighbors=params['n_neighbors'],
                         ref_set=params['ref_set']).fit(X_train)

    def score_samples(self):
        return self.__clf.decision_scores_

    def predict_scores(self, new_samples):
        return self.__clf.decision_function(new_samples)
