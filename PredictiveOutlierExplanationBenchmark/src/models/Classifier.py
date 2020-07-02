import warnings
warnings.filterwarnings("ignore")
from models.classifiers import *
from configpkg.ClassifiersConfig import ClassifiersConfig


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
        predictions_proba = self.__model.predict_proba(X_test)
        predictions_proba = predictions_proba[:, 1]     # Take the probability of a point to be an outlier
        return predictions_proba

    def set_time(self, time):
        self.__time = time

    def get_time(self):
        return self.__time

    def get_params(self):
        return self.__params

    def get_id(self):
        return self.__id

    def get_config(self):
        return self.__classifier_obj

    def to_dict(self):
        classifier_dict = self.__classifier_obj
        classifier_dict[Classifier.TIME_KEY] = self.__time
        return classifier_dict

    @staticmethod
    def time_key():
        return Classifier.TIME_KEY
