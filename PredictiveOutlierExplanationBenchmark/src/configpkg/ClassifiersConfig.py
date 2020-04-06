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
    def list_all_classifiers():
        assert ClassifiersConfig.__classifier_json_obj is not None
        return ClassifiersConfig.__classifier_json_obj

    @staticmethod
    def classifier_key():
        return ClassifiersConfig.__CLASSIFIERS_KEY

    @staticmethod
    def id_key():
        return ClassifiersConfig.__ID_KEY

    @staticmethod
    def params_key():
        return ClassifiersConfig.__PARAMS_KEY

    @staticmethod
    def omit_combinations_key():
        return ClassifiersConfig.__OMIT_COMBINATIONS_KEY

    @staticmethod
    def prime_param_key():
        return ClassifiersConfig.__PRIME_PARAM_KEY

    @staticmethod
    def combs_key():
        return ClassifiersConfig.__COMBS_KEY
