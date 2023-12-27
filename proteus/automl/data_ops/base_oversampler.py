import abc
import pandas as pd
import numpy as np


class BaseOversampler:

    def __init__(self, oversampling_ratio: float):
        assert 0 <= oversampling_ratio < 1, 'oversampling_ratio must be 0 <= and < 1'
        self.oversampling_ratio = oversampling_ratio

    @abc.abstractmethod
    def augment_dataset(
            self,
            X: pd.DataFrame,
            anomaly_scores: np.ndarray,
            anomaly_labels: np.ndarray,
    ):
        raise NotImplementedError("augment_dataset method should be implemented")
