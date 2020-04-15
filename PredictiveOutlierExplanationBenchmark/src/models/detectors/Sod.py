from pyod.models.sod import SOD
import pandas as pd

class Sod:

    def __init__(self):
        self.__clf = None

    def train(self, X_train, params):
        self.__clf = SOD(alpha=params['alpha'], n_neighbors=params['n_neighbors'],
                         ref_set=params['ref_set']).fit(X_train)

    def score_samples(self):
        return self.__clf.decision_scores_

    def predict_scores(self, new_samples):
        if isinstance(new_samples, pd.DataFrame):
            new_samples = new_samples.values
        return self.__clf.decision_function(new_samples)
