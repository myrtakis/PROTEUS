from typing import Callable, List

import numpy as np
import pandas as pd

from itertools import product

from . import BaseAutoML

from proteus.automl.models.classifiers import BaseClassifier
from proteus.automl.models.feature_selection import BaseFeatureSelector
from proteus.automl.data_ops import BaseOversampler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .performance_estimation import BBC


class ProteusAutoML(BaseAutoML):

    def __init__(
            self,
            classifiers: List[BaseClassifier],
            feature_selectors: List[BaseFeatureSelector],
            perf_metric_func: Callable,
            partitions: int,
            explanation_size: int,
            oversampler: BaseOversampler,
            **kwargs
    ):
        super().__init__()
        self.partitions = partitions
        self.explanation_size = explanation_size
        self.classifiers = classifiers
        self.feature_selectors = feature_selectors
        self.perf_metric_func = perf_metric_func
        self.oversampler = oversampler

        self.scaler = None
        self.anomaly_ratio = 0.0

        self.best_clf = None
        self.selected_features = None

        assert all([isinstance(c, BaseClassifier) for c in self.classifiers]), \
            'All classifiers must be of type BaseClassifier.'

        assert all([isinstance(f, BaseFeatureSelector) for f in self.feature_selectors]), \
            'All feature selectors must be of type BaseFeatureSelector.'

        assert isinstance(oversampler, BaseOversampler), 'The oversampler must be of type BaseOversampler'

    def run(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            anomaly_scores: pd.Series
    ):
        y= np.array(y)
        assert len({0, 1}.difference(np.unique(y))) == 0, \
            'The y array must be binary. The 0 corresponds to normal and 1 to anomalous points'

        assert 0 < self.partitions < X.shape[1], f'explanation_size must be 0 < explanation_size < {X.shape[1]}'

        assert 0 < self.partitions < X.shape[0], f'partitions must be 0 < partitions < {X.shape[0]}'

        self.anomaly_ratio = sum(y) / len(y)

        best_clf, best_fsel, perf, predictions_mat = \
            self.__cv(X, y, anomaly_scores, self.classifiers, self.feature_selectors, self.perf_metric_func)
        X, _ = self.__standardize_arrays(X)
        best_fsel.fit(X, y)
        sel_features = best_fsel.feature_coefs(X, dim=self.explanation_size)
        best_clf.fit(X.iloc[:, sel_features], y)
        self.scaler = StandardScaler().fit(X)
        # apply Bootstrap Bias Correction
        bbc = BBC(y, predictions_mat, self.perf_metric_func)
        conservative_performance, _ = bbc.correct_bias()
        self.best_model = best_clf
        self.explanation = sel_features
        self.performance = conservative_performance
        return best_clf, sel_features, conservative_performance

    def predict_new_data(self, X: pd.DataFrame):
        X = self.scaler(X)
        return self.best_clf.predict(X)

    def __cv(self, X, y, anomaly_scores, classifiers, feature_selectors, perf_met):
        print('Selecting the best model')
        learning_methods = list(product(classifiers, feature_selectors))
        num_samples = X.shape[0]
        performances = np.zeros(len(learning_methods))
        predictions_mat = pd.DataFrame(np.full((num_samples, len(learning_methods)), np.NaN))
        splitter = StratifiedKFold(n_splits=self.partitions, shuffle=True)
        for train_ids, test_ids in splitter.split(X, y):
            X_train, X_test = self.__standardize_arrays(X.loc[train_ids], X.loc[test_ids])
            y_train, y_test = y[train_ids], y[test_ids]
            # apply oversampling
            X_train, y_train = self.oversampler.augment_dataset(X_train, y_train, anomaly_scores)
            X_test, y_test = self.oversampler.augment_dataset(X_test, y_test, anomaly_scores)
            for i, (clf, fsel) in enumerate(learning_methods):
                fsel.fit(X_train, y_train)
                sel_features = fsel.feature_coefs(X_train, dim=self.explanation_size)
                if sel_features is None or len(sel_features) == 0:
                    continue
                X_train_reduced = X_train.iloc[:, sel_features]
                X_test_reduced = X_test.iloc[:, sel_features]
                clf.fit(X_train_reduced, y_train)
                predictions = pd.DataFrame(clf.predict(X_test_reduced), index=X_test.index)
                performances[i] += perf_met(y_test, predictions)
                predictions_mat.loc[test_ids, i] = predictions.loc[test_ids].values.flatten()
        performances = np.array(performances) / self.partitions
        best_method = np.argmax(performances)
        best_clf, best_fsel = learning_methods[best_method][0], learning_methods[best_method][1]
        return best_clf, best_fsel, performances[best_method], predictions_mat.dropna(axis=1)

    def __standardize_arrays(self, X_train, X_test=None):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        if X_test is not None:
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        return X_train, X_test
