import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


class SES:

    __sesLib = importr('MXM')

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        if not isinstance(Y_train, pd.Series):
            Y_train = pd.Series(Y_train)
        pandas2ri.activate()
        rpy2_version = rpy2.__version__
        if int(rpy2_version[0:rpy2_version.index('.')]) < 3:
            X_train_r = pandas2ri.py2ri(X_train)
            Y_train_r = pandas2ri.py2ri(Y_train)
        else:
            X_train_r = ro.conversion.py2rpy(X_train)
            Y_train_r = ro.conversion.py2rpy(Y_train)
        ses_object = SES.__sesLib.SES(Y_train_r, X_train_r, max_k=self.__params['max_k'], threshold=self.__params['alpha'])
        selected_vars = np.array(ses_object.slots['selectedVarsOrder'])
        equivalent_features = np.array(ses_object.slots['signatures']) - 1
        if len(selected_vars) == 0:
            return [], None
        else:
            return selected_vars - 1, equivalent_features  # Reduce ids by 1 as R starts counting from 1
