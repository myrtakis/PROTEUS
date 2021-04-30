from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np


class OMP:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.__params['n_nonzero_coefs']).fit(X_train, Y_train)
        if len(np.nonzero(omp.coef_ != 0)[0]) == 0:
            return [], None
        features_sorted = np.argsort(omp.coef_)[::-1]
        zero_coefs = np.nonzero(omp.coef_[features_sorted] == 0)[0]
        selected_features = []
        if len(zero_coefs) > 0:
            selected_features = features_sorted[0:zero_coefs[0]]
        return selected_features, None