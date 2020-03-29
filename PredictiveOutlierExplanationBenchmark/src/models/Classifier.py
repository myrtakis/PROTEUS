from PredictiveOutlierExplanationBenchmark.src.models.classifiers import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import ClassifiersConfig


class Classifier:

    __algorithms = {
        'rf': RandomForest,
        'svm': SVM,
        'knn': KNN
    }

    __PREDICTIONS_KEY = 'predictions'
    __EFFECTIVENESS_KEY = 'effectiveness'
    __TIME_KEY = 'time'

    def __init__(self, classifier_obj):
        assert classifier_obj[ClassifiersConfig.id_key()] in Classifier.__algorithms
        self.__classifier_obj = classifier_obj
        self.__params = classifier_obj[ClassifiersConfig.params_key()]
        self.__id = classifier_obj[ClassifiersConfig.id_key()]
        self.__predictions = None
        self.__time = None
        self.__effectiveness = None
        self.__model = None

    def train(self, X_train, Y_train):
        clf = Classifier.__algorithms[self.__id]
        self.__model = clf(self.__params).train(X_train, Y_train)

    def predict(self, X_test):
        self.__predictions = self.__model.predict(X_test)
        return self.__predictions

    def set_time(self, time):
        self.__time = time

    def set_effectiveness(self, effectiveness):
        self.__effectiveness = effectiveness

    def get_time(self):
        return self.__time

    def get_params(self):
        return self.__params

    def get_predictions(self):
        return self.__predictions

    def get_effectiveness(self):
        return self.__effectiveness

    def to_dict(self):
        self.__classifier_obj[Classifier.__EFFECTIVENESS_KEY] = self.__effectiveness
        self.__classifier_obj[Classifier.__TIME_KEY] = self.__time
        self.__classifier_obj[Classifier.__PREDICTIONS_KEY] = self.__predictions
        return self.__classifier_obj

    @staticmethod
    def predictions_key():
        return Classifier.__PREDICTIONS_KEY

    @staticmethod
    def time_key():
        return Classifier.__TIME_KEY

    @staticmethod
    def effectiveness_key():
        return Classifier.__EFFECTIVENESS_KEY
