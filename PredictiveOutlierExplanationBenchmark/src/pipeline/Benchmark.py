from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.holders.ModelConf import ModelConf
import time
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_all_metrics, calculate_metric, metric_names
import collections
from PredictiveOutlierExplanationBenchmark.src.pipeline.ModelConfigsGen import generate_param_combs
import json


class Benchmark:

    __fsel_key = FeatureSelectionConfig.feature_selection_key()
    __clf_key = ClassifiersConfig.classifier_key()
    __repetition = 1
    __repetition_time = 0

    @staticmethod
    def run(pseudo_samples, dataset):
        print('\n')
        print('Pseudo samples:', pseudo_samples)
        sss = StratifiedShuffleSplit(n_splits=SettingsConfig.get_repetitions(),
                                     test_size=SettingsConfig.get_test_size())
        Benchmark.__repetition = 1
        benchmark_dict = {}
        for train_index, test_index in sss.split(dataset.get_X(), dataset.get_Y()):
            start_time = time.time()
            X_train, X_test = dataset.get_X().iloc[train_index, :], dataset.get_X().iloc[test_index, :]
            Y_train, Y_test = dataset.get_Y()[train_index], dataset.get_Y()[test_index]
            best_models_trained = Benchmark.__cross_validation(X_train, Y_train)
            results = Benchmark.__test_best_models(best_models_trained, X_test, Y_test)
            benchmark_dict[Benchmark.__repetition] = results
            end_time = time.time()
            Benchmark.__repetition_time = round(end_time - start_time, 3)
            Benchmark.__repetition += 1
        best_model_dict = Benchmark.__best_model_dict(benchmark_dict)
        return benchmark_dict, best_model_dict

    @staticmethod
    def __cross_validation(X, Y):
        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(Y))
        skf = StratifiedKFold(n_splits=kfolds, random_state=None, shuffle=True)
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        fold_id = 0
        conf_perfs_per_metric = {}
        total_combs = len(fsel_conf_combs) * len(classifiers_conf_combs)
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            conf_id = 0
            fold_id += 1
            for fsel_conf in fsel_conf_combs:
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                if len(fsel.get_features()) > 0:
                    X_train_new = X_train.iloc[:, fsel.get_features()]
                    X_test_new = X_test.iloc[:, fsel.get_features()]
                    assert X_train_new.shape[1] == len(fsel.get_features())
                    assert X_test_new.shape[1] == len(fsel.get_features())
                for classifier_conf in classifiers_conf_combs:
                    classifier = Classifier(classifier_conf)
                    Benchmark.__console_log(fold_id, fsel, classifier)
                    if len(fsel.get_features()) > 0:
                        classifier.train(X_train_new, Y_train).predict(X_test_new)
                    conf_perfs_per_metric = Benchmark.__update_conf_perfs_per_metric(conf_perfs_per_metric, conf_id,
                                                                                     fsel, classifier, Y_test)
                    conf_id += 1
            assert total_combs == conf_id, str(total_combs) + ' ' + str(conf_id)
        best_confs = Benchmark.__select_best_confs_from_cv(conf_perfs_per_metric, kfolds)
        best_models_trained = Benchmark.__train_best_models(best_confs, X, Y)
        return best_models_trained

    @staticmethod
    def __update_conf_perfs_per_metric(conf_perfs_per_metric, conf_id, fsel, classifier, Y_true):
        fsel_clf_id = fsel.get_id() + '_' + classifier.get_id()
        Y_pred = classifier.get_predictions()
        if Y_pred is None:
            assert len(fsel.get_features()) == 0
            metrics_vals = dict.fromkeys(metric_names(), 0.0)
        else:
            metrics_vals = calculate_all_metrics(Y_true, Y_pred)
        for m_id, val in metrics_vals.items():
            conf_perfs_per_metric.setdefault(m_id, {})
            conf_perfs_per_metric[m_id].setdefault(fsel_clf_id, {})
            if conf_id not in conf_perfs_per_metric[m_id][fsel_clf_id]:
                mc = ModelConf(fsel, classifier, conf_id)
                mc.set_effectiveness(val, m_id, conf_id)
                conf_perfs_per_metric[m_id][fsel_clf_id][conf_id] = mc
            else:
                old_effectiveness = conf_perfs_per_metric[m_id][fsel_clf_id][conf_id].get_effectiveness()
                conf_perfs_per_metric[m_id][fsel_clf_id][conf_id].set_effectiveness(old_effectiveness + val, m_id, conf_id)
        return conf_perfs_per_metric

    @staticmethod
    def __select_best_confs_from_cv(conf_perfs_per_metric, kfolds):
        best_confs_per_metric = {}
        for m_id, metric_data in conf_perfs_per_metric.items():
            best_confs_per_metric.setdefault(m_id, {})
            for fsel_clf_id, conf_data in metric_data.items():
                conf_data_vals = list(conf_data.values())
                sorted_conf_data = sorted(conf_data_vals, key=lambda x: x.get_effectiveness(), reverse=True)
                best_conf = sorted_conf_data[0]
                avg_effectiveness = float(best_conf.get_effectiveness()) / float(kfolds)
                best_conf.set_effectiveness(avg_effectiveness, m_id, best_conf.get_conf_id())
                best_confs_per_metric[m_id][fsel_clf_id] = best_conf
        return best_confs_per_metric

    @staticmethod
    def __train_best_models(best_confs_per_metric, X_train, Y_train):
        print(X_train.shape)
        best_models_trained = {}
        conf_id = 0
        for m_id, metric_data in best_confs_per_metric.items():
            for fsel_clf_id, best_conf in metric_data.items():
                start = time.time()
                fsel = FeatureSelection(best_conf.get_fsel().get_config())
                fsel.run(X_train, Y_train)
                if len(fsel.get_features()) == 0:
                    assert best_conf.get_effectiveness() == 0
                    continue
                end = time.time()
                fsel.set_time(round(end - start, 2))
                classifier = Classifier(best_conf.get_clf().get_config())
                X_train_new = X_train.iloc[:, fsel.get_features()]
                start = time.time()
                classifier.train(X_train_new, Y_train)
                end = time.time()
                classifier.set_time(round(end - start, 2))
                mc = ModelConf(fsel, classifier, conf_id)
                best_models_trained.setdefault(m_id, {})
                best_models_trained[m_id][fsel_clf_id] = mc
                conf_id += 1
        return best_models_trained

    @staticmethod
    def __test_best_models(best_confs_per_metric, X_test, Y_test):
        for m_id, metric_data in best_confs_per_metric.items():
            for fsel_clf_id, conf in metric_data.items():
                X_test_new = X_test.iloc[:, conf.get_fsel().get_features()]
                conf.get_clf().predict(X_test_new)
                Y_pred = conf.get_clf().get_predictions()
                assert Y_pred is not None
                val = calculate_metric(Y_test, Y_pred, m_id)
                conf.set_effectiveness(val, m_id, conf.get_conf_id())
        return best_confs_per_metric

    @staticmethod
    def __best_model_dict(benchmark_dict):
        best_model_per_metric = {}
        for rep, rep_data in benchmark_dict.items():
            for m_id, metric_data in rep_data.items():
                best_model_per_metric.setdefault(m_id, None)
                for fsel_clf_id, conf_data in metric_data.items():
                    if best_model_per_metric[m_id] is None:
                        best_model_per_metric[m_id] = conf_data
                    if best_model_per_metric[m_id].get_effectiveness() < conf_data.get_effectiveness():
                        best_model_per_metric[m_id] = conf_data
                    elif conf_data.get_effectiveness() == best_model_per_metric[m_id].get_effectiveness():
                        if len(conf_data.get_fsel().get_features()) < len(best_model_per_metric[m_id].get_fsel().get_features()):
                            best_model_per_metric[m_id] = conf_data
        return best_model_per_metric

    @staticmethod
    def __console_log(fold_id, fsel, classifier):
        print('\r', 'Repetition', Benchmark.__repetition, '> Fold', fold_id, ':', fsel.get_id(), fsel.get_params(), '|',
              classifier.get_id(), classifier.get_params(), '( Repetition', Benchmark.__repetition - 1, 'needed',
              Benchmark.__repetition_time, 'secs )', end='')

    @staticmethod
    def __get_rarest_class_count(Y):
        return int(min(collections.Counter(Y).values()))

"""
    @staticmethod
    def __cross_validation(X, Y):
        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(Y))
        skf = StratifiedKFold(n_splits=kfolds, random_state=None, shuffle=True)
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        confs_info = {}
        confs_perf = {}
        fold_id = 1
        total_combs = len(fsel_conf_combs) * len(classifiers_conf_combs)
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            conf_id = 0
            for fsel_conf in fsel_conf_combs:
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                if len(fsel.get_features()) > 0:
                    X_train = X_train.iloc[:, fsel.get_features()]
                    X_test = X_test.iloc[:, fsel.get_features()]
                    assert X_train.shape[1] == len(fsel.get_features())
                for classifier_conf in classifiers_conf_combs:
                    classifier = Classifier(classifier_conf)
                    confs_info = Benchmark.__update_confs_info(confs_info, conf_id, fsel, classifier)
                    print('\r', 'Fold', fold_id, ':', fsel.get_id(), fsel.get_params(), '|', classifier.get_id(), classifier.get_params(), end='')
                    if len(fsel.get_features()) == 0:
                        confs_perf = Benchmark.__update_confs_perf(confs_perf, conf_id, Y_test, None)
                    else:
                        classifier.train(X_train, Y_train).predict(X_test)
                        assert classifier.get_predictions() is not None
                        confs_perf = Benchmark.__update_confs_perf(confs_perf, conf_id, Y_test, classifier.get_predictions())
                    conf_id += 1
            assert conf_id == total_combs, str(conf_id) + ' != ' + str(total_combs)
            fold_id += 1
        print()
        best_conf_perfs = Benchmark.__select_best_confs_perf(confs_perf, confs_info, kfolds)
        return Benchmark.__train_best_models(best_conf_perfs, X, Y)

    @staticmethod
    def __train_best_models(best_models_confs, X_train, Y_train):
        # TODO check why with lasso alpha 0.01 we take no feature back in train and how it escaped by the best models
        best_models = {}
        for m_id, metric_data in best_models_confs.items():
            for conf_id, data in metric_data.items():
                start = time.time()
                fsel = FeatureSelection(data[Benchmark.__fsel_key].get_config())
                fsel.run(X_train, Y_train)
                end = time.time()
                fsel.set_time(round(end - start, 2))
                classifier = Classifier(data[Benchmark.__clf_key].get_config())
                assert len(fsel.get_features()) > 0
                X_train = X_train.iloc[:, fsel.get_features()]
                start = time.time()
                classifier.train(X_train, Y_train)
                end = time.time()
                classifier.set_time(round(end - start, 2))
                best_models.setdefault(m_id, {})
                best_models[m_id][conf_id] = {Benchmark.__fsel_key: fsel, Benchmark.__clf_key: classifier}
        return best_models

    @staticmethod
    def __test_best_models(best_models, X_test, Y_test):
        old_results = {}
        for m_id in best_models:
            for conf_id, data in best_models[m_id].items():
                fsel = data[Benchmark.__fsel_key]
                classifier = data[Benchmark.__clf_key]
                X_test = X_test.iloc[:, fsel.get_features()]
                classifier.predict(X_test)
                Y_pred = data[Benchmark.__clf_key].get_predictions()
                val = calculate_metric(Y_test, Y_pred, m_id)
                classifier.set_effectiveness(val)
                old_results.setdefault(m_id, {})
                old_results[m_id][conf_id] = data
                old_results[m_id][conf_id] = {Benchmark.__clf_key: classifier.to_dict(), Benchmark.__fsel_key: fsel.to_dict()}
        return old_results

    @staticmethod
    def __update_confs_perf(confs_perf, conf_id, y_true, y_pred):
        if y_pred is None:
            metrics_perfs = dict.fromkeys(metric_names(), 0.0)
        else:
            metrics_perfs = calculate_all_metrics(y_true, y_pred)
        for m_id, value in metrics_perfs.items():
            if m_id not in confs_perf:
                confs_perf[m_id] = {}
            if conf_id not in confs_perf[m_id]:
                confs_perf[m_id][conf_id] = 0
            confs_perf[m_id][conf_id] += value
        return confs_perf

    @staticmethod
    def __update_confs_info(confs_info, conf_id, fsel, classifier):
        if conf_id in confs_info:
            return confs_info
        confs_info[conf_id] = {Benchmark.__fsel_key: fsel, Benchmark.__clf_key: classifier}
        return confs_info

    @staticmethod
    def __avg_confs_perf(confs_perf, kfolds):
        for m_id in confs_perf:
            for conf_id, value in confs_perf[m_id].items():
                confs_perf[m_id][conf_id] /= kfolds
        return confs_perf

    @staticmethod
    def __select_best_confs_perf(confs_perf, confs_info, kfolds):
        confs_perf = Benchmark.__avg_confs_perf(confs_perf, kfolds)
        best_confs_perf = {}
        for m_id in confs_perf:
            for conf_id, value in confs_perf[m_id].items():
                fsel_clf_id = Benchmark.__get_fsel_clf_id_from_conf(confs_info[conf_id])
                if m_id not in best_confs_perf:
                    best_confs_perf[m_id] = {}
                if fsel_clf_id not in best_confs_perf[m_id]:
                    if value > 0:
                        best_confs_perf[m_id][fsel_clf_id] = (value, conf_id)
                else:
                    curr_max, curr_max_conf_id = best_confs_perf[m_id][fsel_clf_id][0], best_confs_perf[m_id][fsel_clf_id][1]
                    curr_max_features = confs_info[curr_max_conf_id][Benchmark.__fsel_key].get_features()
                    curr_features = confs_info[conf_id][Benchmark.__fsel_key].get_features()
                    if Benchmark.__current_conf_better(value, curr_max, curr_features, curr_max_features):
                        best_confs_perf[m_id][fsel_clf_id] = (value, conf_id)
        for m_id in best_confs_perf:
            for conf_id, info_tuple in best_confs_perf[m_id].items():
                best_confs_perf[m_id][conf_id] = confs_info[info_tuple[1]]
        return best_confs_perf

    @staticmethod
    def __current_conf_better(curr_value, curr_max, curr_conf_features, max_conf_features):
        assert curr_max > 0
        better = True
        if curr_value < curr_max:
            better = False
        elif curr_value == curr_max:
            if len(curr_conf_features) > len(max_conf_features):
                better = False
        return better

    @staticmethod
    def __get_fsel_clf_id_from_conf(conf_info):
        fsel_id = conf_info[Benchmark.__fsel_key].get_id()
        clf_id = conf_info[Benchmark.__clf_key].get_id()
        return fsel_id + '_' + clf_id

    @staticmethod
    def __get_fsel_clf_id_from_objs(fsel, classifier):
        return fsel.get_id() + '_' + classifier.get_id()



"""