from PredictiveOutlierExplanationBenchmark.src.models.detectors.iForest import iForest
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Lof import Lof
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorConfig import DetectorConfig


class Detector:

    def __init__(self):
        self.__detectors_map = {
            'lof': Lof(),
            'iforest': iForest()
        }
        self.__effectiveness = None
        self.__output = None
        self.__labels = []
        assert DetectorConfig.get_id() in self.__detectors_map

    # Base Functions

    def set_effectiveness(self, effectiveness):
        self.__effectiveness = effectiveness

    def set_labels(self, labels):
        self.__labels = labels
        self.__convert_labels_to_int_type()

    def train(self, X_train):
        self.__detectors_map[DetectorConfig.get_id()].train(X_train, DetectorConfig.get_params())

    def score_samples(self):
        self.__output = self.__detectors_map[DetectorConfig.get_id()].score_samples()
        self.__convert_output_to_float_type()
        return self.__output

    def predict(self, X_test):
        self.__output = self.__detectors_map[DetectorConfig.get_id()].predict_scores(X_test)
        self.__convert_output_to_float_type()
        return self.__output

    def get_output(self):
        return self.__output

    def get_effectiveness(self):
        return self.__effectiveness

    def get_labels(self):
        return self.__labels

    def to_dict(self):
        return {
            DetectorConfig.detector_key():
                {
                    DetectorConfig.id_key(): DetectorConfig.get_id(),
                    DetectorConfig.params_key(): DetectorConfig.get_params(),
                    'effectiveness': self.__effectiveness,
                    'labels': self.__labels,
                    'output': self.__output
                }
        }

    # Util Functions

    def __convert_output_to_float_type(self):
        self.__output = [float(x) for x in self.__output]

    def __convert_labels_to_int_type(self):
        self.__labels = [int(x) for x in self.__labels]

    def list_all_detectors(self):
        return list(self.__detectors_map.keys())
