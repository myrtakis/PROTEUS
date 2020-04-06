class FeatureSelectionConfig:

    __FEATURE_SELECTION_KEY = "feature_selection"
    __ID_KEY = "id"
    __PARAMS_KEY = "params"
    __feature_selection_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        FeatureSelectionConfig.__feature_selection_json_obj = config_json_obj[FeatureSelectionConfig.__FEATURE_SELECTION_KEY]

    @staticmethod
    def list_all_feature_selection_algs():
        assert FeatureSelectionConfig.__feature_selection_json_obj is not None
        return FeatureSelectionConfig.__feature_selection_json_obj

    @staticmethod
    def id_key():
        return FeatureSelectionConfig.__ID_KEY

    @staticmethod
    def params_key():
        return FeatureSelectionConfig.__PARAMS_KEY

    @staticmethod
    def feature_selection_key():
        return FeatureSelectionConfig.__FEATURE_SELECTION_KEY
