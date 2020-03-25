class FeatureSelectionConfig:

    def __init__(self):
        self.__FEATURE_SELECTION_KEY = "feature_selection"
        self.__ID_KEY = "id"
        self.__PARAMS_KEY = "params"
        self.__feature_selection_json_obj = None

    def setup(self, feature_selection_json_obj):
        self.__feature_selection_json_obj = feature_selection_json_obj[self.__FEATURE_SELECTION_KEY]
