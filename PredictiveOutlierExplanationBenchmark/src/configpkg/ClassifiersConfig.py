class ClassifiersConfig:

    def __init__(self):
        self.__CLASSIFIERS_KEY = "classifiers"
        self.__ID_KEY = "id"
        self.__PARAMS_KEY = "params"
        self.__OMIT_COMBINATIONS_KEY = "omit_combinations"
        self.__PRIME_PARAM_KEY = "prime_param"
        self.__COMBS_KEY = "combs"
        self.__classifier_json_obj = None

    def setup(self, classifier_json_obj):
        self.__classifier_json_obj = classifier_json_obj[self.__CLASSIFIERS_KEY]

    def get_id(self):
        return self.__classifier_json_obj[self.__ID_KEY]

    def get_params(self):
        return self.__classifier_json_obj[self.__PARAMS_KEY]

    def get_omit_combinations(self):
        return self.__classifier_json_obj[self.__OMIT_COMBINATIONS_KEY]

    def get_prime_param_key(self):
        return self.__PRIME_PARAM_KEY

    def get_combs_key(self):
        return self.__COMBS_KEY
