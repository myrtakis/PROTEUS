import numpy as np
import pandas as pd
from typing import List, Callable
from proteus.automl import BaseAutoML, ProteusAutoML
from sklearn.metrics import roc_auc_score


class ProteusExplainer:

    def __init__(self, automl: BaseAutoML):

        assert isinstance(automl, BaseAutoML), \
            'The automl must be of type BaseAutoML'

        self.automl = automl

        self.explainer_fitted = False

    def explain(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            anomaly_scores: pd.Series
    ):
        self.automl.run(X, y, anomaly_scores)
        self.explainer_fitted = True

    def explain_new_data(self, X):
        assert self.explainer_fitted, 'Run explain method first.'
        return self.automl.predict_new_data(X)
