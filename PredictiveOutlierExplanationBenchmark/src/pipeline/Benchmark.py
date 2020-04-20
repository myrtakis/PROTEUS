from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.holders.ModelConf import ModelConf
import time
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_all_metrics, calculate_metric, \
    metric_names
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
        train_test_indices_dict = {}
        for train_index, test_index in sss.split(dataset.get_X(), dataset.get_Y()):
            start_time = time.time()
            X_train, X_test = dataset.get_X().iloc[train_index, :], dataset.get_X().iloc[test_index, :]
            Y_train, Y_test = dataset.get_Y()[train_index], dataset.get_Y()[test_index]
            best_models_trained = Benchmark.__cross_validation(X_train, Y_train)
            results = Benchmark.__test_best_models(best_models_trained, X_test, Y_test)
            benchmark_dict[Benchmark.__repetition] = results
            end_time = time.time()
            Benchmark.__repetition_time = round(end_time - start_time, 3)
            train_test_indices_dict = Benchmark.__update_train_test_indices_dict(train_test_indices_dict,
                                                                                 train_index, test_index)
            Benchmark.__repetition += 1
        best_model_fs_dict = Benchmark.__best_model_dict(benchmark_dict, fs=True)
        best_model_no_fs_dict = Benchmark.__best_model_dict(benchmark_dict, fs=False)
        merged_best_models = Benchmark.__merge_to_dicts_depth1(best_model_fs_dict, best_model_no_fs_dict)
        return benchmark_dict, merged_best_models, train_test_indices_dict

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
                        classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
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
        Y_pred = classifier.get_predictions_proba()
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
                conf_perfs_per_metric[m_id][fsel_clf_id][conf_id].set_effectiveness(old_effectiveness + val, m_id,
                                                                                    conf_id)
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
        best_models_trained = {}
        conf_id = 0
        for m_id, metric_data in best_confs_per_metric.items():
            for fsel_clf_id, best_conf in metric_data.items():
                start = time.time()
                fsel = FeatureSelection(best_conf.get_fsel().get_config())
                fsel.run(X_train, Y_train)
                if len(fsel.get_features()) == 0:
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
                conf.get_clf().predict_proba(X_test_new)
                Y_pred = conf.get_clf().get_predictions_proba()
                assert Y_pred is not None
                val = calculate_metric(Y_test, Y_pred, m_id)
                conf.set_effectiveness(val, m_id, conf.get_conf_id())
        return best_confs_per_metric

    @staticmethod
    def __best_model_dict(benchmark_dict, fs):
        best_model_per_metric = {}
        for rep, rep_data in benchmark_dict.items():
            for m_id, metric_data in rep_data.items():
                best_model_per_metric.setdefault(m_id, {})
                for fsel_clf_id, conf_data in metric_data.items():
                    if fs is True and 'none' in fsel_clf_id:
                        continue
                    if fs is False and 'none' not in fsel_clf_id:
                        continue
                    if len(best_model_per_metric[m_id]) == 0:
                        best_model_per_metric[m_id] = conf_data
                    if best_model_per_metric[m_id].get_effectiveness() < conf_data.get_effectiveness():
                        best_model_per_metric[m_id] = conf_data
                    elif conf_data.get_effectiveness() == best_model_per_metric[m_id].get_effectiveness():
                        if len(conf_data.get_fsel().get_features()) < len(best_model_per_metric[m_id].get_fsel().get_features()):
                            best_model_per_metric[m_id] = conf_data
        # Refine the key names in dict
        for m_id, best_conf in best_model_per_metric.items():
            fsel_clf_id = best_conf.get_fsel().get_id() + '_' + best_conf.get_clf().get_id()
            best_model_per_metric[m_id] = {fsel_clf_id: best_conf}
        return best_model_per_metric

    @staticmethod
    def __console_log(fold_id, fsel, classifier):
        print('\r', 'Repetition', Benchmark.__repetition, '> Fold', fold_id, ':', fsel.get_id(), fsel.get_params(), '|',
              classifier.get_id(), classifier.get_params(), '( Repetition', Benchmark.__repetition - 1, 'needed',
              Benchmark.__repetition_time, 'secs )', end='')

    @staticmethod
    def __get_rarest_class_count(Y):
        return int(min(collections.Counter(Y).values()))

    @staticmethod
    def __merge_to_dicts_depth1(d1, d2):
        merged_d = {}
        for k in d1:
            merged_d.setdefault(k, {})
            merged_d[k].update(d1[k])
            merged_d[k].update(d2[k])
        return merged_d

    @staticmethod
    def __update_train_test_indices_dict(train_test_indices_dict, train_ind, test_ind):
        train_ind = [int(x) for x in train_ind]
        test_ind = [int(x) for x in test_ind]
        train_test_indices_dict.setdefault(Benchmark.__repetition, {})
        train_test_indices_dict[Benchmark.__repetition]['train_indices'] = train_ind
        train_test_indices_dict[Benchmark.__repetition]['test_indices'] = test_ind
        return train_test_indices_dict
