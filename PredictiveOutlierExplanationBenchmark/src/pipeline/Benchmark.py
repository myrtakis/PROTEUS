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
import pandas as pd
from PredictiveOutlierExplanationBenchmark.src.pipeline.ModelConfigsGen import generate_param_combs
import json
from PredictiveOutlierExplanationBenchmark.src.pipeline.BbcCorrection import BBC


class Benchmark:
    __fsel_key = FeatureSelectionConfig.feature_selection_key()
    __clf_key = ClassifiersConfig.classifier_key()

    @staticmethod
    def run(pseudo_samples, dataset):
        print('\n')
        print('Pseudo samples:', pseudo_samples)
        benchmark_dict = {}
        train_test_indices_dict = {}
        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(dataset.get_Y()))
        assert kfolds > 1, kfolds
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=0)
        no_fs_dict = Benchmark.__cross_validation2(dataset.get_X(), dataset.get_Y(), skf, kfolds, knowledge_discovery=False)
        no_fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(no_fs_dict)

        fs_dict = Benchmark.__cross_validation2(dataset.get_X(), dataset.get_Y(), skf, kfolds, knowledge_discovery=True)
        fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(fs_dict)
        print()
        exit()

        """
        for train_index, test_index in sss.split(dataset.get_X(), dataset.get_Y()):
            start_time = time.time()
            X_train, X_test = dataset.get_X().iloc[train_index, :], dataset.get_X().iloc[test_index, :]
            Y_train, Y_test = dataset.get_Y()[train_index], dataset.get_Y()[test_index]
            best_models_trained = Benchmark.__cross_validation(X_train, Y_train)
            results = Benchmark.__test_best_models(best_models_trained, X_test, Y_test)
            end_time = time.time()
            Benchmark.__repetition_time = round(end_time - start_time, 3)
            train_test_indices_dict = Benchmark.__update_train_test_indices_dict(train_test_indices_dict,
                                                                                 train_index, test_index)
        best_model_fs_dict = Benchmark.__best_model_dict(benchmark_dict, fs=True)
        best_model_no_fs_dict = Benchmark.__best_model_dict(benchmark_dict, fs=False)
        merged_best_models = Benchmark.__merge_to_dicts_depth1(best_model_fs_dict, best_model_no_fs_dict)
        """
        # return benchmark_dict, merged_best_models, train_test_indices_dict

    @staticmethod
    def __cross_validation2(X, Y, skf, folds, knowledge_discovery):
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        fold_id = 0
        conf_data_in_folds = {}
        folds_true_labels = {}
        true_labels = np.array([])
        total_combs = len(fsel_conf_combs) * len(classifiers_conf_combs) - len(classifiers_conf_combs) if knowledge_discovery is True else len(classifiers_conf_combs)
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            conf_id = 0
            fold_id += 1
            folds_true_labels[fold_id] = Y_test
            true_labels = np.concatenate((true_labels, Y_test))
            for fsel_conf in fsel_conf_combs:
                if Benchmark.__omit_fsel(fsel_conf, knowledge_discovery):
                    continue
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                if len(fsel.get_features()) > 0:
                    X_train_new = X_train.iloc[:, fsel.get_features()]
                    X_test_new = X_test.iloc[:, fsel.get_features()]
                    assert X_test_new.shape[1] == len(fsel.get_features())
                for classifier_conf in classifiers_conf_combs:
                    classifier = Classifier(classifier_conf)
                    Benchmark.__console_log(fold_id, fsel, classifier)
                    conf_id += 1
                    conf_data_in_folds.setdefault(conf_id, {})
                    if len(fsel.get_features()) > 0:
                        classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
                        conf_data_in_folds[conf_id][fold_id] = ModelConf(fsel, classifier, conf_id)
            assert total_combs == conf_id, str(total_combs) + ' ' + str(conf_id)
        conf_perfs = Benchmark.__compute_confs_perf_per_metric(conf_data_in_folds, folds_true_labels, folds)
        best_model_per_metric = Benchmark.__select_best_model_per_metric(conf_perfs)
        print('\n', best_model_per_metric)
        best_model_trained_per_metric = Benchmark.__train_best_model_in_all_data(best_model_per_metric, conf_data_in_folds, X, Y)
        predictions_merged = Benchmark.__merge_predictions_from_folds(conf_data_in_folds, folds)
        return {'conf_data_in_folds': conf_data_in_folds,
                'best_model_trained_per_metric': best_model_trained_per_metric,
                'predictions_merged': predictions_merged, 'true_labels': true_labels}

    @staticmethod
    def __merge_predictions_from_folds(conf_data_in_folds, folds):
        conf_data_predictions_mrg = {}
        for c_id, c_data in conf_data_in_folds.items():
            conf_data_predictions_mrg.setdefault(c_id, np.array([], dtype=np.float64))
            assert folds >= len(c_data)
            if len(c_data) < folds:
                continue
            for fold_id in sorted(c_data.keys()):
                predictions = c_data[fold_id].get_clf().get_predictions_proba()
                conf_data_predictions_mrg[c_id] = np.concatenate((conf_data_predictions_mrg[c_id], predictions))
        return conf_data_predictions_mrg

    @staticmethod
    def __compute_confs_perf_per_metric(conf_data_in_folds, folds_true_labels, folds):
        conf_perfs = {}
        for c_id, c_data in conf_data_in_folds.items():
            for f_id in sorted(c_data.keys()):
                metrics_dict = calculate_all_metrics(folds_true_labels[f_id], c_data[f_id].get_clf().get_predictions_proba())
                for m_id, val in metrics_dict.items():
                    conf_perfs.setdefault(m_id, {})
                    conf_perfs[m_id].setdefault(c_id, 0.0)
                    conf_perfs[m_id][c_id] += val
        for m_id, m_data in conf_perfs.items():
            for c_id, perf in m_data.items():
                conf_perfs[m_id][c_id] /= folds
        return conf_perfs

    @staticmethod
    def __select_best_model_per_metric(conf_perfs_per_metric):
        best_model_per_metric = {}
        for m_id, m_data in conf_perfs_per_metric.items():
            best_perf = None
            best_c_id = None
            for c_id, perf in m_data.items():
                if best_perf is None or best_perf < perf:
                    best_perf = perf
                    best_c_id = c_id
            best_model_per_metric[m_id] = {best_c_id: best_perf}
        return best_model_per_metric

    @staticmethod
    def __train_best_model_in_all_data(best_model_per_metric, conf_data_in_folds, X, Y):
        for m_id, m_data in best_model_per_metric.items():
            for best_c_id, c_data in m_data.items():
                conf = conf_data_in_folds[best_c_id][1]  # simply take the configuration of the 1st fold (starting by 1) which is the same for every fold
                fsel = FeatureSelection(conf.get_fsel().get_config())
                start = time.time()
                fsel.run(X, Y)
                end = time.time()
                fsel.set_time(round(end - start, 2))
                assert len(fsel.get_features()) > 0
                clf = Classifier(conf.get_clf().get_config())
                start = time.time()
                clf.train(X, Y)
                end = time.time()
                clf.set_time(round(end - start, 2))
                best_model_per_metric[m_id] = ModelConf(fsel, clf, -1)
        return best_model_per_metric

    @staticmethod
    def __omit_fsel(fsel_conf, knowledge_discovery):
        if knowledge_discovery is True and 'none' in fsel_conf['id'].lower():
            return True
        if knowledge_discovery is False and 'none' not in fsel_conf['id'].lower():
            return True
        return False

    @staticmethod
    def __remove_bias(data_dict):
        for m_id in data_dict['best_model_trained_per_metric']:
            preds = pd.DataFrame(data_dict['predictions_merged']).values
            correct_perf = BBC(data_dict['true_labels'], preds, m_id).correct_bias()
            data_dict['best_model_trained_per_metric'][m_id].set_effectiveness(correct_perf, m_id, -1)
        return data_dict['best_model_trained_per_metric']


    @staticmethod
    def __console_log(fold_id, fsel, classifier):
        print('\r', 'Fold', fold_id, ':', fsel.get_id(), fsel.get_params(), '>',
              classifier.get_id(), classifier.get_params(), end='')

    @staticmethod
    def __get_rarest_class_count(Y):
        return int(min(collections.Counter(Y).values()))


