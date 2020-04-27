import warnings
warnings.filterwarnings("ignore")
from PredictiveOutlierExplanationBenchmark.src.models.classifiers import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import ClassifiersConfig
import numpy as np


class Classifier:

    __algorithms = {
        'rf': RandomForest,
        'svm': SVM,
        'knn': KNN
    }

    PREDICTIONS_PROBA_KEY = 'predictions_proba'
    PREDICTIONS_LABELS_KEY = 'predictions_label'
    TIME_KEY = 'time'

    def __init__(self, classifier_obj):
        assert classifier_obj[ClassifiersConfig.id_key()] in Classifier.__algorithms
        self.__classifier_obj = classifier_obj
        self.__params = classifier_obj[ClassifiersConfig.params_key()]
        self.__id = classifier_obj[ClassifiersConfig.id_key()]
        self.__predictions_proba = None
        self.__preditions_labels = None
        self.__time = None
        self.__model = None

    def __str__(self):
        return self.to_dict()

    def __repr__(self):
        return str(self.to_dict())

    def train(self, X_train, Y_train):
        clf = Classifier.__algorithms[self.__id]
        self.__model = clf(self.__params).train(X_train, Y_train)
        return self

    def predict_proba(self, X_test):
        self.__predictions_proba = self.__model.predict_proba(X_test)
        self.__assign_predictions_labels()
        self.__modidy_predictions_proba()
        return self.__predictions_proba

    def set_time(self, time):
        self.__time = time

    def get_time(self):
        return self.__time

    def get_params(self):
        return self.__params

    def get_id(self):
        return self.__id

    def get_predictions_proba(self):
        return self.__predictions_proba

    def get_predictions_labels(self):
        return self.__preditions_labels

    def get_config(self):
        return self.__classifier_obj

    def to_dict(self):
        classifier_dict = self.__classifier_obj
        classifier_dict[Classifier.TIME_KEY] = self.__time
        classifier_dict[Classifier.PREDICTIONS_PROBA_KEY] = self.__convert_array_to_float_type(self.__predictions_proba)
        classifier_dict[Classifier.PREDICTIONS_LABELS_KEY] = self.__convert_array_to_int_type(self.__preditions_labels)
        return classifier_dict

    def __modidy_predictions_proba(self):
        tmp = np.copy(self.__preditions_labels).astype(np.float)
        class_zero_labels = np.where(tmp == 0)
        class_one_labels = np.where(tmp == 1)
        tmp[class_zero_labels] = np.min(self.__predictions_proba[class_zero_labels], axis=1)
        tmp[class_one_labels] = np.max(self.__predictions_proba[class_one_labels], axis=1)
        self.__predictions_proba = tmp

    def __assign_predictions_labels(self):
        self.__preditions_labels = np.argmax(self.__predictions_proba, axis=1)

    def __convert_array_to_int_type(self, array):
        if array is None:
            return None
        return [int(x) for x in array]

    def __convert_array_to_float_type(self, array):
        if array is None:
            return None
        return [float(x) for x in array]

    @staticmethod
    def time_key():
        return Classifier.TIME_KEY
