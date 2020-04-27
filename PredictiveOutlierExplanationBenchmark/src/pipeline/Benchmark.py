from sklearn.model_selection import StratifiedKFold
from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.holders.ModelConf import ModelConf
import time
import numpy as np
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

        kfolds = min(SettingsConfig.get_kfolds(), Benchmark.__get_rarest_class_count(dataset.get_Y()))
        assert kfolds > 1, kfolds
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=0)

        no_fs_dict = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), skf, kfolds, knowledge_discovery=False)
        no_fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(no_fs_dict)

        fs_dict = Benchmark.__cross_validation(dataset.get_X(), dataset.get_Y(), skf, kfolds, knowledge_discovery=True)
        fs_dict['best_model_trained_per_metric'] = Benchmark.__remove_bias(fs_dict)
        return Benchmark.__make_results(no_fs_dict, fs_dict)

    @staticmethod
    def __cross_validation(X, Y, skf, folds, knowledge_discovery):
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        fold_id = 0
        conf_data_in_folds = {}
        folds_true_labels = {}
        train_test_indices_folds = {}
        true_labels = np.array([])
        total_combs = len(fsel_conf_combs) * len(classifiers_conf_combs) - len(classifiers_conf_combs) if knowledge_discovery is True else len(classifiers_conf_combs)
        start = time.time()
        print('Knowledge Discovery:', knowledge_discovery, ', Total Configs:', total_combs)
        for train_index, test_index in skf.split(X, Y):
            elapsed_time = time.time() - start
            start = elapsed_time
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            conf_id = 0
            fold_id += 1
            train_test_indices_folds[fold_id] = {'train_indices': train_index, 'test_indices': test_index}
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
                    Benchmark.__console_log(fold_id, fsel, classifier, round(elapsed_time, 2))
                    conf_id += 1
                    conf_data_in_folds.setdefault(conf_id, {})
                    if len(fsel.get_features()) > 0:
                        classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
                        conf_data_in_folds[conf_id][fold_id] = ModelConf(fsel, classifier, conf_id)
            assert total_combs == conf_id, str(total_combs) + ' ' + str(conf_id)
        print()
        conf_data_in_folds = Benchmark.__exclude_confs_with_no_selected_features(conf_data_in_folds, folds)
        conf_data_in_folds_small_expl = Benchmark.__exclude_explanations_with_many_features(conf_data_in_folds, knowledge_discovery, Benchmark.__MAX_FEATURES)
        conf_perfs = Benchmark.__compute_confs_perf_per_metric(conf_data_in_folds_small_expl, folds_true_labels, folds)
        best_model_per_metric = Benchmark.__select_best_model_per_metric(conf_perfs)
        best_model_trained_per_metric = Benchmark.__train_best_model_in_all_data(best_model_per_metric, conf_data_in_folds_small_expl, X, Y)
        predictions_merged = Benchmark.__merge_predictions_from_folds(conf_data_in_folds_small_expl, folds)
        return {'best_model_trained_per_metric': best_model_trained_per_metric,
                'predictions_merged': predictions_merged, 'true_labels': true_labels,
                'train_test_indices_folds': train_test_indices_folds,
                'conf_data_in_folds': conf_data_in_folds}

    @staticmethod
    def __exclude_confs_with_no_selected_features(conf_data_in_folds, folds):
        conf_data_in_folds_cleaned = {}
        for c_id, c_data in conf_data_in_folds.items():
            assert folds >= len(c_data)
            if len(c_data) == folds:
                conf_data_in_folds_cleaned[c_id] = c_data
        return conf_data_in_folds_cleaned

    @staticmethod
    def __exclude_explanations_with_many_features(conf_data_in_folds, knowledge_discovery, max_features):
        if knowledge_discovery is False:
            return conf_data_in_folds
        conf_data_in_folds_small_explanations = {}
        while len(conf_data_in_folds_small_explanations) < 2:
            for c_id, c_data in conf_data_in_folds.items():
                feature_num_per_fold = []
                for f_id, f_data in c_data.items():
                    feature_num_per_fold.append(len(f_data.get_fsel().get_features()))
                if np.mean(feature_num_per_fold) <= max_features:
                    conf_data_in_folds_small_explanations[c_id] = c_data
            max_features += 1
        return conf_data_in_folds_small_explanations

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
        for m_id, m_data in best_model_per_metric.items():
            for best_c_id, c_data in m_data.items():
                conf = conf_data_in_folds[best_c_id][1]  # simply take the configuration of the 1st fold (starting by 1) which is the same for every fold
                print('\r', 'Training in all data the', conf.get_fsel().get_config(), '>', conf.get_clf().get_config(), end='')
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
              'was', elapsed_time, 'secs', end='')

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

