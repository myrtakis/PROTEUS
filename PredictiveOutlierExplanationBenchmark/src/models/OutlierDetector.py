import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np


class Detector:

    def __init__(self, detector_obj):
        self.__detectors_map = {
            'lof': {'train': self.__lof_train, 'predict': self.__lof_predict},
            'iforest': {'train': self.__iforest_train, 'predict': self.__iforest_predict}
        }
        assert detector_obj['id'] in self.__detectors_map
        self.__detectorObj = detector_obj

    # Base Functions

    def train(self, X_train):
        train_func = self.__detectors_map[self.__detectorObj['id']]['train']
        train_func(X_train, self.__detectorObj['params'], self.__detectorObj['repetitions'])

    def predict(self, X_test):
        predict_func = self.__detectors_map[self.__detectorObj['id']]['predict']
        return predict_func(X_test)

    # Detectors Train

    def __lof_train(self, X_train, params, repetitions=None):
        self.__clf = LocalOutlierFactor(n_neighbors=params['n_neighbors'], novelty=True,
                                        contamination='auto').fit(X_train)

    def __iforest_train(self, X_train, params, repetitions=1):
        self.__clf = []
        max_samples = min(X_train.shape[0], params['max_samples'])
        warnings.filterwarnings("ignore")
        for i in range(0, repetitions):
            self.__clf.append(IsolationForest(max_samples=max_samples, n_estimators=params['n_estimators'],
                                              behaviour='new', contamination='auto').fit(X_train))

    # Detectors Predict

    def __lof_predict(self, X_test):
        return np.array(self.__clf.score_samples(X_test)) * -1

    def __iforest_predict(self, X_test):
        predictions = None
        for m in self.__clf:
            tmp_arr = np.array(m.score_samples(X_test)) * -1
            if predictions is None:
                predictions = tmp_arr
            else:
                predictions = np.vstack((predictions, tmp_arr)).T
        return np.average(predictions, axis=1)

    # Util Functions

    def list_all_detectors(self):
        return list(self.__detectors_map.keys())
