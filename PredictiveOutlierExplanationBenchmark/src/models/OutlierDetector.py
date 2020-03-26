from PredictiveOutlierExplanationBenchmark.src.models.detectors.iForest import iForest
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Lof import Lof
import numpy as np


class Detector:

    def __init__(self, detector_conf_obj):
        self.__detectors_map = {
            'lof': iForest(),
            'iforest': Lof()
        }
        assert detector_conf_obj['id'] in self.__detectors_map
        self.__detector_conf_obj = detector_conf_obj

    # Base Functions

    def train(self, X_train):
        self.__detectors_map[self.__detector_conf_obj['id']].train(X_train, self.__detector_conf_obj['params'])

    def score_samples(self):
        return self.__detectors_map[self.__detector_conf_obj['id']].score_samples()

    def predict(self, X_test):
        return self.__detectors_map[self.__detector_conf_obj['id']].predict_scores(X_test)

    # Util Functions

    def list_all_detectors(self):
        return list(self.__detectors_map.keys())
