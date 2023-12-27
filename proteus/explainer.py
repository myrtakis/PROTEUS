import pandas as pd
from typing import List, Any
from proteus.automl import BaseAutoML
from proteus.visualization import spider_plot


class ProteusExplainer:

    def __init__(self, automl: BaseAutoML):

        assert isinstance(automl, BaseAutoML), \
            'The automl must be of type BaseAutoML'

        self.automl = automl

        self.explainer_fitted = False

    def fit_explainer(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            anomaly_scores: pd.Series
    ):
        self.automl.run(X, y, anomaly_scores)
        self.explainer_fitted = True

    def get_explanation(self):
        assert self.explainer_fitted, 'Explainer should be fitted first'
        return self.automl.explanation

    def get_final_model(self):
        assert self.explainer_fitted, 'Explainer should be fitted first'
        return self.automl.best_model

    def approximation_quality(self):
        assert self.explainer_fitted, 'Explainer should be fitted first'
        return self.automl.performance

    def explain_new_data(self, X):
        assert self.explainer_fitted, 'Explainer should be fitted first'
        return self.automl.predict_new_data(X)

    def visualize(
            self,
            X: pd.DataFrame,
            sample_ids: List[int],
            explanation_features: List[Any] = None
    ):
        if explanation_features is None:
            explanation_features = self.automl.explanation
        explanation_features = list(map(type(X.columns[0]), list(explanation_features)))
        spider_plot(
            sample_ids_to_plot=sample_ids,
            X=X.loc[:, explanation_features]
        )
