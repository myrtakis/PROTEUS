from pyod.models.sod import SOD
import pandas as pd

class Sod:

    def __init__(self):
        self.__clf = None
        self.__fitted_X = None

    def train(self, X_train, params):
        self.__fitted_X = X_train
        self.__clf = SOD(alpha=params['alpha'], n_neighbors=params['n_neighbors'],
                         ref_set=params['ref_set']).fit(X_train)

    def score_samples(self):
        return self.__clf.decision_scores_

    def predict_scores(self, new_samples):
        if isinstance(new_samples, pd.DataFrame):
            new_samples = new_samples.values
        new_samples_df = pd.DataFrame(new_samples, columns=self.__fitted_X.columns)
        concat_df = pd.concat([self.__fitted_X, new_samples_df], ignore_index=True)
        scores = self.__clf.decision_function(concat_df.values)
        return scores[self.__fitted_X.shape[0]:]
