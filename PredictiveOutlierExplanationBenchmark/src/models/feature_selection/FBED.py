import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd


class FBED:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        if not isinstance(Y_train, pd.Series):
            Y_train = pd.Series(Y_train)
        fbed = rpy2.robjects.r("fbed.reg")
        pandas2ri.activate()
        rpy2_version = rpy2.__version__
        if int(rpy2_version[0:rpy2_version.index('.')]) < 3:
            X_train_r = pandas2ri.py2ri(X_train)
            Y_train_r = pandas2ri.py2ri(Y_train)
        else:
            X_train_r = ro.conversion.py2rpy(X_train)
            Y_train_r = ro.conversion.py2rpy(Y_train)
        fbed_object = fbed(Y_train_r, X_train_r, threshold=self.__params['threshold'], K=self.__params['K'], test='testIndReg')
        selected_vars = np.array(fbed_object[1], dtype=int)[:, 0]
        return selected_vars - 1, None  # Reduce ids by 1 as R starts counting from 1
