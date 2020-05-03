from sklearn.linear_model import Lasso
import numpy as np


class LASSO:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        lasso = Lasso(alpha=self.__params['alpha']).fit(X_train, Y_train)
        if len(np.nonzero(lasso.coef_ != 0)[0]) == 0:
            return [], None
        features_sorted = np.argsort(lasso.coef_)[::-1]
        zero_coefs = np.nonzero(lasso.coef_[features_sorted] == 0)[0]
        selected_features = []
        if len(zero_coefs) > 0:
            selected_features = features_sorted[0:zero_coefs[0]]
        return selected_features, None
