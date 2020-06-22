import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Lasso


class CA_Lasso:

    def __init__(self, dataset_detected_outliers):
        self.dataset = dataset_detected_outliers
        self.reg_strength = 0.35
        if dataset_detected_outliers.get_df().shape[0] < 35:
            self.k = dataset_detected_outliers.get_df().shape[0] / 2
        else:
            self.k = 35
        self.alpha = 0.35

    def run(self):
        local_explanations = {}
        for o in self.dataset.get_outlier_indices():
            inliers, kth_distance = self.inlier_undersampling(o)
            outliers = self.outlier_oversampling(o, inliers.shape[0], kth_distance)
            local_explanations[o] = self.local_explanation(inliers, outliers)
        feature_scores = None
        for o, expl in local_explanations.items():
            if feature_scores is None:
                feature_scores = expl
            else:
                feature_scores += expl
        feature_scores /= len(local_explanations)
        features_ranked = np.argsort(feature_scores)[::-1]
        return local_explanations

    def inlier_undersampling(self, outlier_ind):
        knn = NearestNeighbors(n_neighbors=self.k+1).fit(self.dataset.get_X())
        distances, indices = knn.kneighbors(self.dataset.get_X())
        kth_distance = distances[outlier_ind, -1]
        left_inds = np.delete(np.array(self.dataset.get_X().index), indices[outlier_ind])
        if 2 * self.k > self.dataset.get_X().shape[0]:
            random_inds = left_inds
        else:
            random_inds = np.random.choice(left_inds.shape[0], self.k, replace=False)
        random_inliers = self.dataset.get_X().iloc[random_inds, :]
        outlier_knns = self.dataset.get_X().iloc[indices[outlier_ind, 1:], :]
        inliers = pd.concat([random_inliers, outlier_knns], axis=0)
        inlier_class = pd.concat([inliers.reset_index(drop=True), pd.Series(np.zeros(inliers.shape[0]))], axis=1)
        return inlier_class, kth_distance

    def outlier_oversampling(self, outlier_ind, pseudo_samples_num, kth_distance):
        outlier = self.dataset.get_X().iloc[outlier_ind, :]
        outliers = []
        l = self.alpha * (1/np.sqrt(self.dataset.get_X().shape[1])) * kth_distance
        cov = np.repeat(np.square(l), self.dataset.get_X().shape[1])
        for i in range(pseudo_samples_num-1):
            outliers.append(np.random.normal(outlier, cov))
        outliers.append(outlier.values)
        outlier_class = pd.concat([pd.DataFrame(np.array(outliers)), pd.Series(np.ones(len(outliers)))], axis=1)
        return outlier_class

    def local_explanation(self, inliers, outliers):
        df = pd.DataFrame(np.concatenate((inliers.values, outliers.values), axis=0))
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
        lasso = Lasso(alpha=self.reg_strength).fit(X, Y)
        return lasso.coef_
