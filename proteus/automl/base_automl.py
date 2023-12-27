import abc

import pandas as pd

from typing import Callable, List

from proteus.automl.models.classifiers import BaseClassifier
from proteus.automl.models.feature_selection import BaseFeatureSelector


class BaseAutoML:

    def __init__(self):
        self.best_model     = None
        self.explanation    = None
        self.performance    = None

    @abc.abstractmethod
    def run(self,
            X: pd.DataFrame,
            y: pd.Series,
            anomaly_scores: pd.Series
    ):
        raise NotImplementedError("run method should be implemented")

    @abc.abstractmethod
    def predict_new_data(
            self,
            X: pd.DataFrame
    ):
        raise NotImplementedError("predict_new_data method should be implemented")