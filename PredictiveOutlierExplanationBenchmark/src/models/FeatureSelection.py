from models.feature_selection import *
from configpkg.FeatureSelectionConfig import FeatureSelectionConfig


class FeatureSelection:

    __algorithms = {
        'none': None,
        'explanation': None,
        'ses': SES,
        'lasso': LASSO,
        'fbed': FBED,
        'omp': OMP
    }

    FEATURES_KEY = 'features'
    EQUIVALENT_FEATURES_KEY = 'equivalent_features'
    TIME_KEY = 'time'

    def __init__(self, feature_selection_obj):
        assert feature_selection_obj[FeatureSelectionConfig.id_key()] in FeatureSelection.__algorithms
        self.__feature_selection_obj = feature_selection_obj
        self.__id = feature_selection_obj[FeatureSelectionConfig.id_key()]
        self.__params = feature_selection_obj[FeatureSelectionConfig.params_key()]
        self.__full_selector = self.__id == 'none'
        self.__is_explanation = self.__id == 'explanation'
        self.__features = None
        self.__equivalent_features = None
        self.__initial_features_num = None
        self.__time = None

    def __str__(self):
        return self.to_dict()

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        if not self.__class__ == self.__class__:
            return False
        if len(self.__params.keys()) != len(other.__params.keys()):
            return False
        for k in self.__params:
            if self.__params[k] != other.__params[k]:
                return False
        return True

    def run(self, X_train, Y_train):
        self.__initial_features_num = X_train.shape[1]
        if not self.__full_selector or self.__is_explanation:
            fsel = FeatureSelection.__algorithms[self.__id]
            self.__features, self.__equivalent_features = fsel(self.__params).run(X_train, Y_train)
            self.__convert_features_to_int_type()
            self.__convert_equiv_features_to_int_type()
        return self

    def set_time(self, time):
        self.__time = time

    def set_features(self, features):
        assert features is not None and len(features) > 0
        self.__features = features

    def get_features(self):
        if self.__full_selector:
            assert self.__initial_features_num > 0
            return list(range(self.__initial_features_num))
        else:
            return self.__features

    def get_equivalent_predictive_features(self):
        return self.__equivalent_features

    def get_time(self):
        return self.__time

    def get_params(self):
        return self.__params

    def get_id(self):
        return self.__id

    def get_config(self):
        return self.__feature_selection_obj

    def to_dict(self):
        fsel_as_dict = self.__feature_selection_obj
        fsel_as_dict[FeatureSelection.FEATURES_KEY] = list(self.get_features())
        if self.__equivalent_features is not None:
            fsel_as_dict[FeatureSelection.EQUIVALENT_FEATURES_KEY] = list(self.__equivalent_features)
        else:
            fsel_as_dict[FeatureSelection.EQUIVALENT_FEATURES_KEY] = ''
        fsel_as_dict[FeatureSelection.TIME_KEY] = self.__time
        return fsel_as_dict

    def __convert_features_to_int_type(self):
        if len(self.__features) > 0:
            self.__features = [int(x) for x in self.__features]

    def __convert_equiv_features_to_int_type(self):
        if self.__equivalent_features is not None:
            self.__equivalent_features = [[int(x) for x in lst] for lst in self.__equivalent_features]
