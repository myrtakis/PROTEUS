import warnings
from sklearn.ensemble import IsolationForest
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.models.detectors.base import BaseDetector


class iForest(BaseDetector):

    def __init__(self):
        super().__init__()
        self.__clf = None
        self.__X_train = None
        self.__train_repetitions = 10

    def train(self, X_train, params):
        self.__X_train = X_train
        self.__clf = []
        max_samples = min(X_train.shape[0], params['max_samples'])
        warnings.filterwarnings("ignore")
        for i in range(0, self.__train_repetitions):
            self.__clf.append(IsolationForest(max_samples=max_samples, n_estimators=params['n_estimators'],
                                              behaviour='new', contamination='auto').fit(X_train))

    def score_samples(self):
        scores = None
        for m in self.__clf:
            tmp_arr = np.array(m.score_samples(self.__X_train)) * -1
            if scores is None:
                scores = tmp_arr
            else:
                scores = np.vstack((scores, tmp_arr))
        scores = scores.T
        return np.average(scores, axis=1)

    def predict_scores(self, new_samples):
        predictions = None
        for m in self.__clf:
            tmp_arr = np.array(m.score_samples(new_samples)) * -1
            if predictions is None:
                predictions = tmp_arr
            else:
                predictions = np.vstack((predictions, tmp_arr))
        predictions = predictions.T
        return np.average(predictions, axis=1)

    def get_explanation(self):
        return None

    def calculate_explanation(self, outlier_ids):
        return None

    def convert_to_global_explanation(self):
        return None

    def is_explainable(self):
        return False
