from PredictiveOutlierExplanationBenchmark.src.models.feature_selection import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.FeatureSelectionConfig import FeatureSelectionConfig

class FeatureSelection:

    __algorithms = {
        "none": None,
        "ses": SES,
        "lasso": LASSO
    }

    __FEATURES_KEY = 'features'
    __EQUIVALENT_FEATURES_KEY = 'equivalent_features'
    __TIME_KEY = 'time'

    def __init__(self, feature_selection_obj):
        assert feature_selection_obj[FeatureSelectionConfig.id_key()] in FeatureSelection.__algorithms
        self.__feature_selection_obj = feature_selection_obj
        self.__id = feature_selection_obj[FeatureSelectionConfig.id_key()]
        self.__params = feature_selection_obj[FeatureSelectionConfig.params_key()]
        self.__features = None
        self.__equivalent_features = None
        self.__time = None

    def run(self, X_train, Y_train):
        fsel = FeatureSelection.__algorithms[self.__id]
        self.__features, self.__equivalent_features = fsel(self.__params).run(X_train, Y_train)

    def set_time(self, time):
        self.__time = time

    def get_features(self):
        return self.__features

    def get_equivalent_predictive_features(self):
        return self.__equivalent_features

    def get_time(self):
        return self.__time

    def to_dict(self):
        self.__feature_selection_obj[FeatureSelection.__FEATURES_KEY] = self.__features
        if self.__equivalent_features is not None:
            self.__feature_selection_obj[FeatureSelection.__EQUIVALENT_FEATURES_KEY] = self.__equivalent_features
        else:
            self.__feature_selection_obj[FeatureSelection.__EQUIVALENT_FEATURES_KEY] = ''
        self.__feature_selection_obj[FeatureSelection.__TIME_KEY] = self.__time
        return self.__feature_selection_obj
