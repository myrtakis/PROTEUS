from PredictiveOutlierExplanationBenchmark.src.models.classifiers import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import ClassifiersConfig


class Classifier:

    __algorithms = {
        'rf': RandomForest,
        'svm': SVM,
        'knn': KNN
    }

    PREDICTIONS_KEY = 'predictions'
    TIME_KEY = 'time'

    def __init__(self, classifier_obj):
        assert classifier_obj[ClassifiersConfig.id_key()] in Classifier.__algorithms
        self.__classifier_obj = classifier_obj
        self.__params = classifier_obj[ClassifiersConfig.params_key()]
        self.__id = classifier_obj[ClassifiersConfig.id_key()]
        self.__predictions = None
        self.__time = None
        self.__model = None

    def __str__(self):
        return self.to_dict()

    def train(self, X_train, Y_train):
        clf = Classifier.__algorithms[self.__id]
        self.__model = clf(self.__params).train(X_train, Y_train)
        return self

    def predict(self, X_test):
        self.__predictions = self.__model.predict(X_test)
        self.__convert_predictions_to_int_type()
        return self.__predictions

    def set_time(self, time):
        self.__time = time

    def get_time(self):
        return self.__time

    def get_params(self):
        return self.__params

    def get_id(self):
        return self.__id

    def get_predictions(self):
        return self.__predictions

    def get_config(self):
        return self.__classifier_obj

    def to_dict(self):
        classifier_dict = self.__classifier_obj
        classifier_dict[Classifier.TIME_KEY] = self.__time
        classifier_dict[Classifier.PREDICTIONS_KEY] = list(self.__predictions)
        return classifier_dict

    def __convert_predictions_to_int_type(self):
        self.__predictions = [int(x) for x in self.__predictions]

    @staticmethod
    def predictions_key():
        return Classifier.PREDICTIONS_KEY

    @staticmethod
    def time_key():
        return Classifier.TIME_KEY
