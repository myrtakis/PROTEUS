import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
from sklearn.linear_model import Lasso


sesLib = importr('MXM')


class FeatureSelection:

    def __init__(self, feature_selection_obj):
        self.__feature_selection = {
            "none":     self.__run_none,
            "ses":      self.__run_ses,
            "lasso":    self.__run_lasso
        }
        assert feature_selection_obj['id'] in self.__feature_selection
        self.__feature_selection_obj = feature_selection_obj
        self.__features = None
        self.__equal_features = None

    # Base Functions

    def run_feature_selection(self, X_train, Y_train):
        feature_selection_func = self.__feature_selection[self.__feature_selection_obj['id']]
        self.__features = feature_selection_func(X_train, Y_train, self.__feature_selection_obj['params'])

    # Feature Selection Models

    def __run_none(self, X_train, Y_train, params):
        return np.array(range(0, X_train.shape[1]))

    def __run_ses(self, X_train, Y_train, params):
        pandas2ri.activate()
        rpy2_version = rpy2.__version__
        if int(rpy2_version[0:rpy2_version.index('.')]) < 3:
            X_train_r = pandas2ri.py2ri(X_train)
            Y_train_r = pandas2ri.py2ri(Y_train)
        else:
            X_train_r = ro.conversion.py2rpy(X_train)
            Y_train_r = ro.conversion.py2rpy(Y_train)
        ses_object = sesLib.SES(Y_train_r, X_train_r, max_k=params['max_k'], threshold=params['alpha'])
        selected_vars = np.array(ses_object.slots['selectedVars'])
        self.__equal_features = np.array(ses_object.slots['signatures']) - 1
        return selected_vars - 1  # Reduce ids by 1 as R starts counting from 1

    def __run_lasso(self, X_train, Y_train, params):
        lasso = Lasso(alpha=params['alpha']).fit(X_train, Y_train)
        return np.where(lasso.coef_ != 0)[0]

    # Getter Functions

    def get_features(self):
        return self.__features

    def get_equal_predictive_features(self):
        return self.__equal_features
