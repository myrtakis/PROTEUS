import numpy as np
from PredictiveOutlierExplanationBenchmark.src.models.detectors.base import BaseDetector


class LODA(BaseDetector):

    def __init__(self):
        super().__init__()
        self.projections_ = None
        self.histograms_ = None
        self.limits_ = None
        self.n_bins = None
        self.random_cuts = None
        self.X = None

    def train(self, X_train, params):
        n_components = X_train.shape[1]
        n_nonzero_components = np.sqrt(n_components)
        n_zero_components = n_components - np.int(n_nonzero_components)
        self.X = X_train
        self.random_cuts = params['n_random_cuts']
        self.n_bins = params['n_bins']
        self.projections_ = np.random.randn(self.random_cuts, n_components)
        self.histograms_ = np.zeros((self.random_cuts, self.n_bins))
        self.limits_ = np.zeros((self.random_cuts, self.n_bins + 1))
        for i in range(self.random_cuts):
            rands = np.random.permutation(n_components)[:n_zero_components]
            self.projections_[i, rands] = 0.
            projected_data = self.projections_[i, :].dot(X_train.T)
            self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                projected_data, bins=self.n_bins, density=False)
            self.histograms_[i, :] += 1e-12
            self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

    def score_samples(self):
        pred_scores = np.zeros([self.X.shape[0], 1])
        for i in range(self.random_cuts):
            projected_data = self.projections_[i, :].dot(self.X.T)
            inds = np.searchsorted(self.limits_[i, :self.n_bins - 1], projected_data, side='left')
            pred_scores[:, 0] += -np.log(self.histograms_[i, inds])
        pred_scores = np.concatenate(pred_scores).ravel()
        return pred_scores / self.random_cuts

    def predict_scores(self, new_samples):
        pass

    def get_explanation(self):
        pass