class ClassifiersConfig:

    def __init__(self):
        self.__CLASSIFIERS_KEY = "classifiers"
        self.__ID_KEY = "id"
        self.__PARAMS_KEY = "params"
        self.__classifier_json_obj = None

    def setup(self, classifier_json_obj):
        self.__classifier_json_obj = classifier_json_obj[self.__CLASSIFIERS_KEY]

