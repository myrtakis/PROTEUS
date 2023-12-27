import pandas as pd
import numpy as np
import sklearn

from proteus import ProteusExplainer, ProteusAutoML
from proteus import BaseClassifier, BaseFeatureSelector
from proteus import ProteusOversampler

from sklearn.neighbors import LocalOutlierFactor

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import OrthogonalMatchingPursuit


class SkClassifier(BaseClassifier):

    def __init__(self, model):
        super().__init__(model)

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SkFeatureSelector(BaseFeatureSelector):

    def __init__(self, model):
        super().__init__(model)

    def fit(self, X, y):
        self.model.fit(X, y)

    def feature_coefs(self, X, dim):
        selected_features = None
        coefs = self.model.coef_
        coefs = np.abs(coefs)
        non_zero_coefs = np.nonzero(coefs != 0)[0]
        if len(non_zero_coefs):
            selected_features = np.argsort(coefs)[::-1][:dim]
        return selected_features


if __name__ == '__main__':

    df = pd.read_csv('../datasets/diabetes.csv')
    X = df.drop(columns=['is_anomaly'])

    # run detector

    # select the anomalous samples
    anomaly_threshold = 0.01

    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X)
    scores = -lof.score_samples(X)

    # Binarize anomaly scores
    anom_ids = np.argsort(scores)[::-1][:int(np.floor(len(scores) * anomaly_threshold))]
    is_anomaly = np.zeros(len(scores))
    is_anomaly[anom_ids] = 1

    # Proteus parmeters
    explanation_size = 5

    # Init classification models

    lr_models = [
        SkClassifier(LogisticRegression(C=c)) for c in [1.0, 0.8, 0.5]
    ]

    svm_models = [
        SkClassifier(SVC(kernel=k, gamma=g, C=c, probability=True))
        for (k, g, c) in zip(['rbf', 'linear'], ['scale', 'auto'], [1.0, 0.8, 0.5])
    ]

    rf_models = [
        SkClassifier(RandomForestClassifier(n_estimators=t, criterion=c))
        for (t, c) in zip([200, 400, 600], ['gini', 'entropy'])
    ]

    # Init feature selectors

    omp_fsels = [SkFeatureSelector(OrthogonalMatchingPursuit(n_nonzero_coefs=explanation_size))]

    def detector_fn(X):
        return -lof.score_samples(X)

    # Setup oversampler
    oversampler = ProteusOversampler(
        oversampling_ratio=0.2,
        detector_predict_func=detector_fn
    )

    # Setup AutoML
    automl = ProteusAutoML(
        classifiers=[*lr_models, *svm_models, *rf_models],
        feature_selectors=omp_fsels,
        perf_metric_func=sklearn.metrics.roc_auc_score,
        partitions=5,
        explanation_size=explanation_size,
        oversampler=oversampler
    )

    proteus = ProteusExplainer(automl=automl)

    # Run proteus

    proteus.fit_explainer(X, is_anomaly, scores)

    # visualize

    anomalies = anom_ids[:2]
    print('Plotting Anomalies')
    proteus.visualize(X, anomalies)


