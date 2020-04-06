from PredictiveOutlierExplanationBenchmark.src.models.detectors.iForest import iForest
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Lof import Lof
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorConfig import DetectorConfig


class Detector:

    def __init__(self):
        self.__detectors_map = {
            'lof': Lof(),
            'iforest': iForest()
        }
        self.__output = None
        assert DetectorConfig.get_id() in self.__detectors_map

    # Base Functions

    def train(self, X_train):
        self.__detectors_map[DetectorConfig.get_id()].train(X_train, DetectorConfig.get_params())

    def score_samples(self):
        self.__output = self.__detectors_map[DetectorConfig.get_id()].score_samples()
        return self.__output

    def predict(self, X_test):
        self.__output = self.__detectors_map[DetectorConfig.get_id()].predict_scores(X_test)
        return self.__output

    def get_output(self):
        return self.__output

    # Util Functions

    def list_all_detectors(self):
        return list(self.__detectors_map.keys())
