import pandas as pd

from . import BaseOversampler
import numpy as np
from typing import Callable, List, Iterable


class ProteusOversampler(BaseOversampler):

    def __init__(
            self,
            oversampling_ratio: float,
            detector_predict_func: Callable[[Iterable], List[float]]
    ):
        super().__init__(oversampling_ratio)
        self.detector_predict_func = detector_predict_func

    def augment_dataset(
            self,
            X: pd.DataFrame,
            anomaly_labels: np.ndarray,
            anomaly_scores: np.ndarray,
    ):
        print('Augmenting the dataset')
        X_aug = X.copy()
        anomaly_labels_aug = anomaly_labels.tolist()

        anomaly_ids = np.where(anomaly_labels == 1)[0]
        score_thresh = min(anomaly_scores[anomaly_ids])
        pseudo_samples_per_anom = int(np.ceil(self.oversampling_ratio * len(X) / len(anomaly_ids)))

        pseudo_samples = pd.DataFrame()

        for a_id in anomaly_ids:
            o_sample = X_aug.iloc[a_id, :].values
            for ps_sample in range(pseudo_samples_per_anom):
                pseudo_sample = self.__gaussian_noise(X_aug, o_sample, score_thresh)
                pseudo_samples = pd.concat([pseudo_samples, pd.DataFrame(pseudo_sample)], axis=1)
                anomaly_labels_aug.append(1)
        pseudo_samples = pseudo_samples.T
        pseudo_samples.columns = X_aug.columns
        pseudo_samples.index = [-1] * len(pseudo_samples)
        X_aug = pd.concat([X_aug, pseudo_samples])
        return X_aug, anomaly_labels_aug

    def __gaussian_noise(self, X, o_sample, threshold):
        alpha = 0.1
        iters = 0
        dim = X.shape[1]
        while True:
            mu = np.zeros(dim)
            sigma = np.ones(dim)
            noise = np.random.normal(mu, alpha * sigma, dim)
            pseudo_sample = o_sample + noise
            if self.detector_predict_func(np.array([pseudo_sample])) >= threshold:
                return pseudo_sample
            if iters == 100:
                alpha += 0.05
                iters = 0
            iters += 1
