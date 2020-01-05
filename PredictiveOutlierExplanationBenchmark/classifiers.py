import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor


na = 'na'


def train_classifier(classifier_id, params, sel_variables, X_train, Y_train):
    train_classifiers_map = {
        'iforest': train_iforest,
        'rf': train_random_forest,
        'svm': train_svm
    }
    X_train_mod = X_train.iloc[:, sel_variables]
    return train_classifiers_map[classifier_id](params, X_train_mod, Y_train)


def test_classifier(classifier_id, model, sel_variables, X_test):
    test_classifiers_map = {
        'iforest': test_iforest,
        'rf': test_random_forest,
        'svm': test_svm
    }
    X_test_mod = X_test.iloc[:, sel_variables]
    return test_classifiers_map[classifier_id](model, X_test_mod)


# TRAIN MODELS #


def train_iforest(params, X_train, Y_train):
    rep = 1    # produce 10 different models due to model randomness
    clfs = []
    max_samples = min(X_train.shape[0], params['max_samples'])
    for i in range(0, rep):
        clfs.append(IsolationForest(max_samples=max_samples, n_estimators=params['n_estimators'],
                                    behaviour='new', contamination='auto').fit(X_train))
    return clfs


def train_lof(params, X_train, Y_train):
    return [LocalOutlierFactor(n_neighbors=params['n_neighbors'], novelty=True, contamination='auto')]


def train_random_forest(params, X_train, Y_train):
    return [RandomForestClassifier(n_estimators=params['n_estimators'], min_samples_leaf=params['min_samples_leaf'],
                                   criterion=params['criterion']).fit(X_train, Y_train)]


def train_svm(params, X_train, Y_train):
    if str(params['gamma']).lower() == na:
        params['gamma'] = 'auto'
    if str(params['degree']).lower() == na:
        params['degree'] = 0
    return [SVC(gamma=params['gamma'], kernel=params['kernel'], C=params['C'], degree=params['degree']).fit(X_train, Y_train)]


# TEST MODELS #


def test_iforest(models, X_test):
    predictions = []
    for m in models:
        predictions.append(np.array(m.score_samples(X_test)) * -1)  # reverse the scores because the lowest, the more abnormal
    return predictions


def test_random_forest(model, X_test):
    return [model[0].predict(X_test)]


def test_svm(model, X_test):
    return [model[0].predict(X_test)]


def test_lof(model, X_test):
    return [np.array(model.score_samples(X_test)) * -1]
