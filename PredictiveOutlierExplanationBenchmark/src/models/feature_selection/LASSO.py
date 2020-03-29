from sklearn.linear_model import Lasso
import numpy as np


class LASSO:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        lasso = Lasso(alpha=self.__params['alpha']).fit(X_train, Y_train)
        return np.where(lasso.coef_ != 0)[0], None
