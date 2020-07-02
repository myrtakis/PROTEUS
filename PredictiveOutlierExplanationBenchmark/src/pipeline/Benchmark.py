from sklearn.model_selection import StratifiedKFold
from configpkg import *
from models.FeatureSelection import FeatureSelection
from models.Classifier import Classifier
from holders.ModelConf import ModelConf
import time
import numpy as np
from utils.Logger import Logger
from utils.metrics import calculate_all_metrics
import collections
import pandas as pd
from pipeline.ModelConfigsGen import generate_param_combs
from pipeline.BbcCorrection import BBC
from pathlib import Path
from utils.shared_names import FileNames
import json
from collections import OrderedDict


class Benchmark:
    __fsel_key = FeatureSelectionConfig.feature_selection_key()
    __clf_key = ClassifiersConfig.classifier_key()
    __MAX_FEATURES = 10
    __REPETITIONS = 2
    __output_dir = None

    __select_features_by_topk = True

    @staticmethod
    def run(dataset_kind, pseudo_samples, dataset, output_dir):
        Benchmark.__output_dir = output_dir
        print('----------\n')
        print('Kind of dataset:', dataset_kind, ', Pseudo samples:', pseudo_samples)

        Logger.log('==================\nPseudo Samples: ' + str(pseudo_samples))

        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(dataset))
        assert kfolds > 1, kfolds

        conf_perfs_total = {'no_fs': {}, 'fs': {}}
        conf_info_total = {'no_fs': {}, 'fs': {}}
        predictions_total = {'no_fs': [], 'fs': []}

        for r in range(Benchmark.__REPETITIONS):
            skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
            folds_inds = Benchmark.__indices_in_folds(dataset, skf)
            inds_test_ordered = Benchmark.__merge_inds(folds_inds)

            ind_folder = Path(Benchmark.__output_dir, FileNames.indices_folder)
            ind_folder.mkdir(parents=True, exist_ok=True)
            with open(Path(ind_folder, 'repetition' + str(r) + '.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(Benchmark.__folds_inds_to_list(folds_inds), indent=4, separators=(',', ': '), ensure_ascii=False))

            cv_data = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), folds_inds, False, r)
            conf_perfs_total['no_fs'] = Benchmark.__update_perfs(conf_perfs_total['no_fs'], cv_data['conf_perfs'])
            predictions_total['no_fs'].append(Benchmark.__order_predictions(cv_data['predictions'], inds_test_ordered))
            if len(conf_info_total['no_fs']) == 0:
                conf_info_total['no_fs'] = cv_data['conf_info']

            cv_data = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), folds_inds, True, r)
            conf_perfs_total['fs'] = Benchmark.__update_perfs(conf_perfs_total['fs'], cv_data['conf_perfs'])
            predictions_total['fs'].append(Benchmark.__order_predictions(cv_data['predictions'], inds_test_ordered))
            if len(conf_info_total['fs']) == 0:
                conf_info_total['fs'] = cv_data['conf_info']

        best_models_perfs = {
            'no_fs': Benchmark.__select_best_model_per_metric(conf_perfs_total['no_fs']),
            'fs': Benchmark.__select_best_model_per_metric(conf_perfs_total['fs'])
        }

        best_models_trained = {
            'no_fs': Benchmark.__train_best_model_in_all_data(best_models_perfs['no_fs'], conf_info_total['no_fs'],
                                                              dataset.get_X(), dataset.get_Y(), False),
            'fs': Benchmark.__train_best_model_in_all_data(best_models_perfs['fs'], conf_info_total['fs'],
                                                           dataset.get_X(), dataset.get_Y(), True)
        }

        # Performance Estimation

        best_models_trained = {
            'no_fs': Benchmark.__remove_bias(predictions_total['no_fs'], dataset.get_Y(), best_models_trained['no_fs']),
            'fs': Benchmark.__remove_bias(predictions_total['fs'], dataset.get_Y(), best_models_trained['fs'])
        }


        print()

        # return Benchmark.__make_results(no_fs_dict, fs_dict)

    @ staticmethod
    def __cross_validation(X, Y, folds_inds, knowledge_discovery, repetition):
        true_labels, true_labels_per_fold = Benchmark.__folds_true_labels(Y, folds_inds)
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        fsel_in_folds = Benchmark.__run_feature_selection(X, Y, folds_inds, fsel_conf_combs, knowledge_discovery)
        conf_info, predictions = Benchmark.__run_classifiers(X, Y, folds_inds, classifiers_conf_combs,
                                                                      fsel_in_folds, knowledge_discovery)
        print('Run Classifiers: completed')


        conf_perfs, conf_preds = Benchmark.__compute_confs_perf_per_metric_pooled(predictions, true_labels_per_fold)

        Benchmark.__save(conf_perfs, [Benchmark.__output_dir, FileNames.configurations_folder,
                                     FileNames.configurations_perfs_folder], repetition)

        Benchmark.__save(conf_preds, [Benchmark.__output_dir, FileNames.predictions_folder], repetition,
                         lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

        Benchmark.__save(conf_info, [Benchmark.__output_dir, FileNames.configurations_folder,
                                     FileNames.configurations_info_folder],
                         repetition, lambda o: o.to_dict() if isinstance(o, ModelConf) else o)

        Logger.log('Predictions merged successfully')
        print()
        return {
            'conf_info': conf_info,
            'conf_perfs': conf_perfs,
            'predictions': conf_preds
        }

    @staticmethod
    def __save(data, folders_array, repetition, func=lambda o: o):
        path_to_folder = Path(*folders_array)
        path_to_folder.mkdir(parents=True, exist_ok=True)
        output_file = Path(path_to_folder, 'repetition' + str(repetition) + '.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, default=func, indent=4, separators=(',', ': '), ensure_ascii=False))

    @ staticmethod
    def __run_classifiers(X, Y, folds_inds, clf_conf_combs, fsel_in_folds, knowledge_discovery):
        conf_info = {}
        predictions = {}
        elapsed_time = 0.0
        total_combs = len(fsel_in_folds[next(iter(fsel_in_folds))]) * len(clf_conf_combs) if knowledge_discovery is True else len(clf_conf_combs)
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
                    predictions.setdefault(conf_id, {})
                    classifier = Classifier(clf_conf)
                    Logger.log('Train and predict clf ' + str(conf_id) + '/' + str(total_combs) + ': ' + str(
                        classifier.to_dict()))
                    Benchmark.__console_log(fold_id, conf_id+1, total_combs, fsel, classifier, elapsed_time)
                    predictions_proba = classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
                    predictions[conf_id][fold_id] = predictions_proba
                    Logger.log('Classifier train and predict completed')
                    Logger.log(classifier.to_dict())
                    conf_info[conf_id]= ModelConf(fsel, classifier, conf_id)
                    conf_id += 1
            elapsed_time = time.time() - start
        print()
        return conf_info, predictions

    @ staticmethod
    def __run_feature_selection(X, Y, folds_inds, fsel_conf_combs, knowledge_discovery):
        fsel_fold_dict = OrderedDict()
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
                    print('\r', 'Fold', fold_id, '> Running fsel:', fsel_conf, end='')
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
        if Benchmark.__select_features_by_topk is True:
            fsel_fold_dict_cleaned = Benchmark.__select_top_k_features(fsel_fold_dict_cleaned, knowledge_discovery)
        else:
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
    def __indices_in_folds(dataset, skf):
        folds_inds = OrderedDict()
        fold_id = 1
        if dataset.contains_pseudo_samples():
            X = dataset.get_X()[:dataset.last_original_sample_index()]
            Y = dataset.get_Y()[:dataset.last_original_sample_index()]
        else:
            X = dataset.get_X()
            Y = dataset.get_Y()
        for train_inds, test_inds in skf.split(X, Y):
            if dataset.contains_pseudo_samples():
                ps_indices_per_outlier = dataset.get_pseudo_sample_indices_per_outlier()
                train_inds = Benchmark.__add_pseudo_samples_inds(ps_indices_per_outlier, train_inds)
                test_inds = Benchmark.__add_pseudo_samples_inds(ps_indices_per_outlier, test_inds)
            folds_inds[fold_id] = {'train_indices': train_inds, 'test_indices': test_inds}
            fold_id += 1
        return folds_inds

    @staticmethod
    def __add_pseudo_samples_inds(pseudo_samples_inds_per_outlier, inds):
        assert pseudo_samples_inds_per_outlier is not None
        common_inds = set(pseudo_samples_inds_per_outlier.keys()).intersection(inds)
        assert len(common_inds) > 0
        for ind in common_inds:
            ps_samples_range = pseudo_samples_inds_per_outlier[ind]
            ps_samples_inds = np.arange(ps_samples_range[0], ps_samples_range[1])
            inds = np.concatenate((inds, ps_samples_inds))
        return inds


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

    @ staticmethod
    def __select_top_k_features(fsel_in_folds, knowledge_discovery):
        if knowledge_discovery is False:
            return fsel_in_folds
        fsel_fold_dict_topk = {}
        for c_id, c_data in fsel_in_folds.items():
            for f_id, fsel in c_data.items():
                if len(fsel.get_features()) > Benchmark.__MAX_FEATURES:
                    fsel.set_features(fsel.get_features()[0:Benchmark.__MAX_FEATURES])
            fsel_fold_dict_topk[c_id] = c_data
        return fsel_fold_dict_topk

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
    def __merge_inds(folds_inds):
        inds = np.array([], dtype=int)
        for f_id in folds_inds:
            inds = np.concatenate((inds, folds_inds[f_id]['test_indices']))
        return inds

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
    def __compute_confs_perf_per_metric_pooled(predictions, true_labels_per_fold):
        conf_perfs = {}
        conf_preds = {}
        true_labels = np.array([], dtype=int)
        for f_id in true_labels_per_fold:
            true_labels = np.concatenate((true_labels, true_labels_per_fold[f_id]))
            for conf_id in predictions:
                conf_preds.setdefault(conf_id, np.array([], dtype=float))
                conf_preds[conf_id] = np.concatenate((conf_preds[conf_id], predictions[conf_id][f_id]))
        for conf_id, preds in conf_preds.items():
            metrics_dict = calculate_all_metrics(true_labels, preds)
            for m_id, val in metrics_dict.items():
                conf_perfs.setdefault(m_id, {})
                conf_perfs[m_id].setdefault(conf_id, 0.0)
                conf_perfs[m_id][conf_id] += val
        return conf_perfs, conf_preds

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
    def __train_best_model_in_all_data(best_model_per_metric, conf_data_in_folds, X, Y, knowledge_discovery):
        Logger.log('\n****Inside the training of best model in all data')
        for m_id, m_data in best_model_per_metric.items():
            for best_c_id in m_data:
                conf = conf_data_in_folds[best_c_id]
                Logger.log('Metric: ' + m_id)
                print('\rMetric', m_id, ': Training in all data the', conf.get_fsel().get_config(), '>', conf.get_clf().get_config(), end='')
                Logger.log('Run fsel: ' + str(conf.get_fsel().get_config()))
                fsel = FeatureSelection(conf.get_fsel().get_config())
                start = time.time()
                fsel.run(X, Y)
                Logger.log('Fsel run successfully')
                end = time.time()
                if knowledge_discovery is True and Benchmark.__select_features_by_topk:
                    fsel.set_features(fsel.get_features()[0:Benchmark.__MAX_FEATURES])
                fsel.set_time(round(end - start, 2))
                assert len(fsel.get_features()) > 0
                Logger.log('Run clf: ' + str(conf.get_clf().get_config()))
                X_new = X.iloc[:, fsel.get_features()]
                clf = Classifier(conf.get_clf().get_config())
                start = time.time()
                clf.train(X_new, Y)
                Logger.log('Classifier trained successfully')
                end = time.time()
                clf.set_time(round(end - start, 2))
                best_model_per_metric[m_id] = ModelConf(fsel, clf, conf.get_conf_id())
                Logger.log('Classifier trained successfully')
        print()
        return best_model_per_metric

    @staticmethod
    def __update_perfs(old_perfs, new_perfs):
        for m_id, metric_perfs in new_perfs.items():
            for conf_id in metric_perfs:
                old_perf = 0.0 if len(old_perfs) == 0 else old_perfs[m_id][conf_id]
                new_perfs[m_id][conf_id] += old_perf
        return new_perfs

    @staticmethod
    def __order_predictions(predictions, indices):
        return pd.DataFrame(predictions, index=indices).sort_index().values

    @staticmethod
    def __folds_inds_to_list(folds_inds):
        for f_id, data in folds_inds.items():
            for ind_type, inds in data.items():
                folds_inds[f_id][ind_type] = inds.tolist()
        return folds_inds

    @staticmethod
    def __omit_fsel(fsel_conf, knowledge_discovery):
        if knowledge_discovery is True and 'none' in fsel_conf['id'].lower():
            return True
        if knowledge_discovery is False and 'none' not in fsel_conf['id'].lower():
            return True
        return False

    @staticmethod
    def __remove_bias(predictions, true_labels, best_models):
        print()
        for m_id in best_models:
            perf, ci = BBC(true_labels, predictions, m_id).correct_bias()
            best_models[m_id].set_effectiveness(perf, m_id, best_models[m_id].get_conf_id())
            best_models[m_id].set_confidence_intervals(ci, best_models[m_id].get_conf_id())
        print()
        return best_models

    @staticmethod
    def __console_log(fold_id, conf_id, total_confs, fsel, classifier, elapsed_time):
        print('\r', 'Fold', fold_id, ':', conf_id, '/', total_confs, fsel.get_id(), fsel.get_params(), '>',
              classifier.get_id(), classifier.get_params(), 'Time for fold', fold_id-1,
              'was', round(elapsed_time, 2), 'secs', end='')

    @staticmethod
    def __get_rarest_class_count(dataset):
        if not dataset.contains_pseudo_samples():
            assert dataset.get_pseudo_sample_indices_per_outlier() is None
            return int(min(collections.Counter(dataset.get_Y()).values()))
        else:
            assert dataset.last_original_sample_index() is not None
            assert dataset.get_pseudo_sample_indices_per_outlier() is not None
            return int(min(collections.Counter(dataset.get_Y()[:dataset.last_original_sample_index()]).values()))

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

