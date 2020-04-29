from sklearn.model_selection import StratifiedKFold
from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.holders.ModelConf import ModelConf
import time
import numpy as np

from PredictiveOutlierExplanationBenchmark.src.utils.Logger import Logger
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_all_metrics
import collections
import pandas as pd
from PredictiveOutlierExplanationBenchmark.src.pipeline.ModelConfigsGen import generate_param_combs
from PredictiveOutlierExplanationBenchmark.src.pipeline.BbcCorrection import BBC


class Benchmark:
    __fsel_key = FeatureSelectionConfig.feature_selection_key()
    __clf_key = ClassifiersConfig.classifier_key()
    __MAX_FEATURES = 10

    @staticmethod
    def run(pseudo_samples, dataset):
        print('----------\n')
        print('Pseudo samples:', pseudo_samples)

        Logger.log('==================\nPseudo Samples: ' + str(pseudo_samples))

        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(dataset.get_Y()))
        assert kfolds > 1, kfolds
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=0)

        no_fs_dict = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), skf, knowledge_discovery=False)
        no_fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(no_fs_dict)

        fs_dict = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), skf, knowledge_discovery=True)
        fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(fs_dict)
        return Benchmark.__make_results(no_fs_dict, fs_dict)

    @ staticmethod
    def __cross_validation(X, Y, skf, knowledge_discovery):
        folds_inds = Benchmark.__indices_in_folds(X, Y, skf)
        folds = len(folds_inds.keys())
        true_labels, folds_true_labels = Benchmark.__folds_true_labels(Y, folds_inds)
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        fsel_in_folds = Benchmark.__run_feature_selection(X, Y, folds_inds, fsel_conf_combs, knowledge_discovery)
        conf_data_in_folds = Benchmark.__run_classifiers(X, Y, folds_inds, classifiers_conf_combs,
                                                         fsel_in_folds, knowledge_discovery)
        print('Run Classifiers completed')

        conf_perfs = Benchmark.__compute_confs_perf_per_metric(conf_data_in_folds, folds_true_labels, folds)
        best_model_per_metric = Benchmark.__select_best_model_per_metric(conf_perfs)
        best_model_trained_per_metric = Benchmark.__train_best_model_in_all_data(best_model_per_metric,
                                                                                 conf_data_in_folds, X, Y)
        Logger.log('Best model trained successfully')
        predictions_merged = Benchmark.__merge_predictions_from_folds(conf_data_in_folds, folds)
        Logger.log('Predictions merged successfully')
        print()
        return {'best_model_trained_per_metric': best_model_trained_per_metric,
                'predictions_merged': predictions_merged, 'true_labels': true_labels,
                'train_test_indices_folds': folds_inds,
                'conf_data_in_folds': conf_data_in_folds}

    @ staticmethod
    def __run_classifiers(X, Y, folds_inds, clf_conf_combs, fsel_in_folds, knowledge_discovery):
        conf_data_in_folds = {}
        elapsed_time = 0.0
        total_combs = len(fsel_in_folds) * len(clf_conf_combs) - len(
            clf_conf_combs) if knowledge_discovery is True else len(clf_conf_combs)
        Logger.log('===========\nRun Classifiers\n')
        for fold_id, inds in folds_inds.items():
            start = time.time()
            train_inds = inds['train_indices']
            test_inds = inds['test_indices']
            Logger.log('-----------\nFold ' + str(fold_id))
            Logger.log('train indices: ' + str(list(train_inds)))
            Logger.log('test indices: ' + str(list(test_inds)))
            assert not np.array_equal(train_inds, test_inds)
            X_train, X_test = X.iloc[train_inds, :], X.iloc[test_inds, :]
            Y_train, Y_test = Y[train_inds], Y[test_inds]
            conf_id = 0
            for fsel in fsel_in_folds[fold_id]:
                Logger.log('Reading fsel: ' + str(fsel.to_dict()) + '\n')
                X_train_new = X_train.iloc[:, fsel.get_features()]
                X_test_new = X_test.iloc[:, fsel.get_features()]
                assert 0 < len(fsel.get_features()) == X_test_new.shape[1]
                for clf_conf in clf_conf_combs:
                    conf_data_in_folds.setdefault(conf_id, {})
                    classifier = Classifier(clf_conf)
                    Logger.log('Train and predict clf ' + str(conf_id) + '/' + str(total_combs) + ': ' + str(
                        classifier.to_dict()))
                    Benchmark.__console_log(fold_id, fsel, classifier, elapsed_time)
                    classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
                    Logger.log('Classifier train and predict completed')
                    Logger.log(classifier.to_dict())
                    conf_data_in_folds[conf_id][fold_id] = ModelConf(fsel, classifier, conf_id)
                    conf_id += 1
            elapsed_time = time.time() - start
        print()
        return conf_data_in_folds

    @ staticmethod
    def __run_feature_selection(X, Y, folds_inds, fsel_conf_combs, knowledge_discovery):
        fsel_fold_dict = {}
        for fold_id, inds in folds_inds.items():
            train_inds = inds['train_indices']
            test_inds = inds['test_indices']
            assert not np.array_equal(train_inds, test_inds)
            X_train, X_test = X.iloc[train_inds, :], X.iloc[test_inds, :]
            Y_train, Y_test = Y[train_inds], Y[test_inds]
            conf_id = 0
            for fsel_conf in fsel_conf_combs:
                if Benchmark.__omit_fsel(fsel_conf, knowledge_discovery):
                    continue
                Logger.log('\nRun fsel: ' + str(fsel_conf))
                if knowledge_discovery is True:
                    print('\r', 'Running fsel:', fsel_conf, end='')
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                if knowledge_discovery is True:
                    print(' Completed', end='')
                Logger.log('Feature Selection Completed\n')
                fsel_fold_dict.setdefault(conf_id, {})
                if len(fsel.get_features()) > 0:
                    fsel_fold_dict[conf_id].setdefault(fold_id, [])
                    fsel_fold_dict[conf_id][fold_id] = fsel
                conf_id += 1
        fsel_fold_dict_cleaned = Benchmark.__clean_invalid_feature_selection(fsel_fold_dict, len(folds_inds.keys()))
        fsel_fold_dict_cleaned = Benchmark.__exclude_explanations_with_many_features(fsel_fold_dict_cleaned,
                                                                                   knowledge_discovery,
                                                                                   Benchmark.__MAX_FEATURES)
        fsel_fold_restructured = Benchmark.__restructure_fsel_dict(fsel_fold_dict_cleaned, len(folds_inds.keys()))
        print()
        return fsel_fold_restructured

    @ staticmethod
    def __clean_invalid_feature_selection(fsel_fold_dict, folds):
        fsel_fold_dict_cleaned = {}
        for conf_id, c_data in fsel_fold_dict.items():
            assert len(c_data.keys()) <= folds
            if len(c_data.keys()) < folds:
                continue
            Benchmark.__check_if_fsels_are_the_same(list(c_data.values()))
            fsel_fold_dict_cleaned[conf_id] = c_data
        return fsel_fold_dict_cleaned

    @ staticmethod
    def __restructure_fsel_dict(fsel_fold_dict, folds):
        fsel_dict_mod = {}
        for conf_id, c_data in fsel_fold_dict.items():
            assert len(c_data.keys()) == folds
            for fold_id, fsel in c_data.items():
                fsel_dict_mod.setdefault(fold_id, [])
                fsel_dict_mod[fold_id].append(fsel)
        return fsel_dict_mod

    @ staticmethod
    def __check_if_fsels_are_the_same(fsel_arr):
        for i in range(len(fsel_arr) - 1):
            for j in range(i+1, len(fsel_arr)):
                assert fsel_arr[i] == fsel_arr[j]

    @ staticmethod
    def __indices_in_folds(X, Y, skf):
        folds_inds = {}
        fold_id = 1
        for train_index, test_index in skf.split(X, Y):
            folds_inds[fold_id] = {'train_indices': train_index, 'test_indices': test_index}
            fold_id += 1
        return folds_inds

    @ staticmethod
    def __folds_true_labels(Y, folds_inds):
        folds_true_labels = {}
        true_labels = np.array([])
        for fold_id, inds in folds_inds.items():
            true_labels = np.concatenate((true_labels, Y[inds['test_indices']]))
            folds_true_labels[fold_id] = Y[inds['test_indices']]
        return true_labels, folds_true_labels

    @staticmethod
    def __exclude_explanations_with_many_features(fsel_in_folds, knowledge_discovery, max_features):
        if knowledge_discovery is False:
            return fsel_in_folds
        fsel_in_folds_small_expl = {}
        while len(fsel_in_folds_small_expl) < 2:
            for c_id, c_data in fsel_in_folds.items():
                feature_num_per_fold = []
                for f_id, fsel in c_data.items():
                    feature_num_per_fold.append(len(fsel.get_features()))
                if np.mean(feature_num_per_fold) <= max_features:
                    fsel_in_folds_small_expl[c_id] = c_data
            max_features += 1
        return fsel_in_folds_small_expl

    @staticmethod
    def __merge_predictions_from_folds(conf_data_in_folds, folds):
        conf_data_predictions_mrg = {}
        for c_id, c_data in conf_data_in_folds.items():
            conf_data_predictions_mrg.setdefault(c_id, np.array([], dtype=np.float64))
            assert len(c_data) == folds
            for fold_id in sorted(c_data.keys()):
                predictions = c_data[fold_id].get_clf().get_predictions_proba()
                conf_data_predictions_mrg[c_id] = np.concatenate((conf_data_predictions_mrg[c_id], predictions))
        return conf_data_predictions_mrg

    @staticmethod
    def __compute_confs_perf_per_metric(conf_data_in_folds, folds_true_labels, folds):
        conf_perfs = {}
        for c_id, c_data in conf_data_in_folds.items():
            for f_id in sorted(c_data.keys()):
                assert len(c_data) == folds
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
        Logger.log('\n****Inside the training of best model in all data')
        for m_id, m_data in best_model_per_metric.items():
            for best_c_id, c_data in m_data.items():
                conf = conf_data_in_folds[best_c_id][1]  # simply take the configuration of the 1st fold (starting by 1) which is the same for every fold
                Logger.log('Metric: ' + m_id)
                print('\r', 'Training in all data the', conf.get_fsel().get_config(), '>', conf.get_clf().get_config(), end='')
                Logger.log('Run fsel: ' + str(conf.get_fsel().get_config()))
                fsel = FeatureSelection(conf.get_fsel().get_config())
                start = time.time()
                fsel.run(X, Y)
                Logger.log('Fsel run successfully')
                end = time.time()
                fsel.set_time(round(end - start, 2))
                assert len(fsel.get_features()) > 0
                Logger.log('Run clf: ' + str(conf.get_clf().get_config()))
                clf = Classifier(conf.get_clf().get_config())
                start = time.time()
                clf.train(X, Y)
                Logger.log('Classifier trained successfully')
                end = time.time()
                clf.set_time(round(end - start, 2))
                best_model_per_metric[m_id] = ModelConf(fsel, clf, -1)
                Logger.log('Classifier trained successfully')
        print()
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
        print()
        return data_dict['best_model_trained_per_metric']

    @staticmethod
    def __console_log(fold_id, fsel, classifier, elapsed_time):
        print('\r', 'Fold', fold_id, ':', fsel.get_id(), fsel.get_params(), '>',
              classifier.get_id(), classifier.get_params(), 'Time for fold', fold_id-1,
              'was', round(elapsed_time, 2), 'secs', end='')

    @staticmethod
    def __get_rarest_class_count(Y):
        return int(min(collections.Counter(Y).values()))

    @staticmethod
    def __make_results(no_fs_dict, fs_dict):
        return {
            'no_fs': {
                'best_model_trained_per_metric': no_fs_dict['best_model_trained_per_metric'],
                'conf_data_in_folds': no_fs_dict['conf_data_in_folds'],
                'train_test_indices_folds': no_fs_dict['train_test_indices_folds']
            },
            'fs': {
                'best_model_trained_per_metric': fs_dict['best_model_trained_per_metric'],
                'conf_data_in_folds': fs_dict['conf_data_in_folds'],
                'train_test_indices_folds': fs_dict['train_test_indices_folds']
            }
        }

