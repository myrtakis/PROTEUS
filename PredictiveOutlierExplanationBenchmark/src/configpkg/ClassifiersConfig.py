class ClassifiersConfig:

    __CLASSIFIERS_KEY = "classifiers"
    __ID_KEY = "id"
    __PARAMS_KEY = "params"
    __OMIT_COMBINATIONS_KEY = "omit_combinations"
    __PRIME_PARAM_KEY = "prime_param"
    __COMBS_KEY = "combs"
    __classifier_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        ClassifiersConfig.__classifier_json_obj = config_json_obj[ClassifiersConfig.__CLASSIFIERS_KEY]

    @staticmethod
    def get_id():
        return ClassifiersConfig.__classifier_json_obj[ClassifiersConfig.__ID_KEY]

    @staticmethod
    def get_params():
        return ClassifiersConfig.__classifier_json_obj[ClassifiersConfig.__PARAMS_KEY]

    @staticmethod
    def get_omit_combinations():
        return ClassifiersConfig.__classifier_json_obj[ClassifiersConfig.__OMIT_COMBINATIONS_KEY]

    @staticmethod
    def get_prime_param_key():
        return ClassifiersConfig.__PRIME_PARAM_KEY

    @staticmethod
    def get_combs_key():
        return ClassifiersConfig.__COMBS_KEY
