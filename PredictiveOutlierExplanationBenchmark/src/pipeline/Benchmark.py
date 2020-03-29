from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models import *
import time
import numpy as np


class Benchmark:

    @staticmethod
    def run(dataset):
        sss = StratifiedShuffleSplit(n_splits=SettingsConfig.get_repetitions(), test_size=SettingsConfig.get_test_size())
        counter = 0
        for train_index, test_index in sss.split(dataset.get_X(), dataset.get_Y()):
            start_time = time.time()
            print('\nRepetition = ', counter)
            X_train, X_test = dataset.get_X().iloc[train_index, :], dataset.get_X().iloc[test_index, :]
            Y_train, Y_test = dataset.get_Y()[train_index], dataset.get_Y()[test_index]
            Benchmark.__cross_validation(X_train, Y_train)
            exit()

    @staticmethod
    def __cross_validation(X, Y):
        # kfolds = min(SettingsConfig.get_kfolds(), get_rarest_class_count(Y)) # todo na to kanw me df.is_anomaly.count..
        # skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)

        fsel_conf = {'id': 'ses', 'params': {'max_k': 2, 'alpha': 0.1}}
        classifier_conf = {'id': 'rf', 'params': {'n_estimators': 100, 'min_samples_leaf': 2, 'criterion': 'entropy'}}
        fsel = FeatureSelection(fsel_conf)
        fsel.run(X, Y)
        classifier = Classifier(classifier_conf)
        classifier.train(X.iloc[1:100, fsel.get_features()], Y[1:100])
        classifier.predict(X.iloc[100: 130, fsel.get_features()])
        print(classifier.get_predictions())
