from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
from sklearn.linear_model import Lasso


sesLib = importr('MXM')


def run_var_selection(var_sel_id, params, X_train, Y_train):
    var_selection_map = {
        "none": run_none,
        "ses": run_ses,
        "lasso": run_lasso
    }
    return var_selection_map[var_sel_id](X_train, Y_train, params)


def run_none(X_train, Y_train, params):
    return np.array(range(0, X_train.shape[1]))


def run_ses(X_train, Y_train, params):
    pandas2ri.activate()
    X_train_r = ro.conversion.py2rpy(X_train)
    Y_train_r = ro.conversion.py2rpy(Y_train)
    sesObject = sesLib.SES(Y_train_r, X_train_r, max_k=params['max_k'], threshold=params['alpha'])
    selectedVars = sesObject.slots['selectedVars']
    return selectedVars - 1     # Reduce ids by 1 as R starts counting from 1


def run_lasso(X_train, Y_train, params):
    lasso = Lasso(alpha=params['alpha']).fit(X_train, Y_train)
    return np.where(lasso.coef_ != 0)[0]
