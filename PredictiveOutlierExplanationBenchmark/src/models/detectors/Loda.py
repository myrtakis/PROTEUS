from collections import OrderedDict
from operator import itemgetter
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.models.detectors.base import BaseDetector
from scipy.stats import ttest_ind


class LODA(BaseDetector):

    def __init__(self):
        super().__init__()
        self.projections_ = None
        self.histograms_ = None
        self.limits_ = None
        self.n_bins = None
        self.random_cuts = None
        self.X = None
        self.isfitted = False
        self.explanation = None

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
        self.isfitted = True

    def score_samples(self):
        assert self.isfitted
        return self.__predict(self.X)

    def predict_scores(self, new_samples):
        assert self.isfitted
        return self.__predict(new_samples)

    def __predict(self, X):
        pred_scores = np.zeros([X.shape[0], 1])
        for i in range(self.random_cuts):
            projected_data = self.projections_[i, :].dot(X.T)
            inds = np.searchsorted(self.limits_[i, :self.n_bins - 1], projected_data, side='left')
            pred_scores[:, 0] += -np.log(self.histograms_[i, inds])
        pred_scores = np.concatenate(pred_scores).ravel()
        return pred_scores / self.random_cuts

    def calculate_explanation(self, outlier_ids):
        assert self.isfitted
        features_importance = {}
        for o_id in outlier_ids:
            features_importance.setdefault(o_id, {})
            for f_id in range(self.X.shape[1]):
                left_part, right_part = self.__feature_partitions(f_id)
                if len(left_part) < 2 or len(right_part) < 2:
                    continue
                outlier = self.X.iloc[o_id, :]
                lp_scores = self.__partition_scores(left_part, outlier)
                rp_scores = self.__partition_scores(right_part, outlier)
                _, pval = ttest_ind(lp_scores, rp_scores)
                features_importance[o_id][f_id] = pval
            features_importance[o_id] = OrderedDict(sorted(features_importance[o_id].items(), key=itemgetter(1)))
        self.explanation = features_importance
        return features_importance

    def __partition_scores(self, partition, outlier):
        assert len(partition) > 0
        partition_scores = []
        for p_id in partition:
            projected_data = self.projections_[p_id, :].dot(outlier.T)
            inds = np.searchsorted(self.limits_[p_id, :self.n_bins - 1], projected_data, side='left')
            partition_scores.append(-np.log(self.histograms_[p_id, inds]))
        return partition_scores

    def __feature_partitions(self, f_id):
        left_partition = []
        right_partition = []
        for i in range(self.projections_.shape[0]):
            if self.projections_[i, f_id] != 0:
                left_partition.append(i)
            else:
                right_partition.append(i)
        return left_partition, right_partition

    def get_explanation(self):
        return self.explanation

    def is_explainable(self):
        return True
