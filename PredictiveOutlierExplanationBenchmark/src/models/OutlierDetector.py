from PredictiveOutlierExplanationBenchmark.src.models.detectors.iForest import iForest
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Lof import Lof
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Sod import Sod
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorsConfig import DetectorConfig


class Detector:

    __detectors_map = {
            'lof': Lof,
            'iforest': iForest,
            'sod': Sod
        }

    @staticmethod
    def list_all_detectors():
        return list(Detector.__detectors_map.keys())

    @staticmethod
    def init_detectors():
        detectors_array = []
        for det_conf in DetectorConfig.get_detectors_confs():
            detector_id = det_conf[DetectorConfig.id_key()]
            detector_params = det_conf[DetectorConfig.params_key()]
            detector_obj = Detector.__detectors_map[detector_id]()
            det = Detector(detector_obj, detector_id, detector_params)
            detectors_array.append(det)
        return detectors_array

    def __init__(self, detector, det_id, params):
        self.__detector = detector
        self.__params = params
        self.__det_id = det_id
        self.__effectiveness = None
        self.__scores_in_train = []
        self.__scores_in_predict = []

    # Base Functions

    def set_effectiveness(self, effectiveness):
        self.__effectiveness = effectiveness

    def train(self, X_train):
        self.__detector.train(X_train, self.__params)
        self.__scores_in_train = self.__detector.score_samples()
        return self

    def predict(self, X_test):
        self.__scores_in_predict = self.__detector.predict_scores(X_test)
        return self.__scores_in_predict

    def get_effectiveness(self):
        return self.__effectiveness

    def get_scores_in_train(self):
        return self.__scores_in_train

    def get_scores_in_predict(self):
        return self.__scores_in_predict

    def get_id(self):
        return self.__det_id

    def to_dict(self):
        return {
            self.__det_id:
                {
                    DetectorConfig.id_key(): self.__det_id,
                    DetectorConfig.params_key(): self.__params,
                    'effectiveness': self.__effectiveness,
                    'scores_in_train': self.__convert_output_to_float_type(self.__scores_in_train),
                    'scores_in_predict': self.__convert_output_to_float_type(self.__scores_in_predict)
                }
        }

    # Util Functions

    def __convert_output_to_float_type(self, scores):
        return [float(x) for x in scores]

