import time
from collections import OrderedDict
import numpy as np
import pandas as pd
from pipeline.automl.automl_utils import save
from holders.ModelConf import ModelConf
from models import Classifier, FeatureSelection
from utils.metrics import calculate_all_metrics
from utils.shared_names import FileNames
import pipeline.automl.automl_constants as automlconsts


class CV_Classification:

    def __init__(self, knowledge_discovery, clf_configs, output_dir, is_explanation):
        self.__clf_configs = clf_configs
        self.__output_dir = output_dir
        self.__knowledge_discovery = knowledge_discovery
        self.__is_explanation = is_explanation

    def run(self, folds_inds_reps, fsel_objs, dataset):
        perfs_dict, predictions_ordered, confs_info = self.__repeated_cv(folds_inds_reps, fsel_objs, dataset)
        best_model_perfs = CV_Classification.__select_best_model_per_metric(perfs_dict)
        print('\nRun Classifiers: completed')
        best_models_trained = self.__train_best_model_in_all_data(best_model_perfs, confs_info,
                                                                  dataset.get_X(), dataset.get_Y())
        return best_models_trained, predictions_ordered

    def __repeated_cv(self, folds_inds_reps, fsel_objs, dataset):
        perfs_dict = {}
        confs_info = None
        predictions_ordered = []
        for rep, folds_inds in folds_inds_reps.items():
            true_labels, true_labels_per_fold = CV_Classification.__folds_true_labels(dataset.get_Y(), folds_inds)
            test_inds_merged = CV_Classification.__merge_test_inds(folds_inds)
            confs_info, raw_predictions = self.__cross_validate_classifiers(dataset.get_X(), dataset.get_Y(),
                                                                           folds_inds, fsel_objs[rep], rep)
            conf_perfs, conf_preds_merged = CV_Classification.__compute_confs_perf_per_metric_pooled(raw_predictions,
                                                                                                     true_labels_per_fold)
            perfs_dict = CV_Classification.__update_perfs(perfs_dict, conf_perfs)
            predictions_ordered.append(CV_Classification.__order_predictions(conf_preds_merged, test_inds_merged))
            save(conf_perfs, [self.__output_dir, FileNames.configurations_folder,
                              FileNames.configurations_perfs_folder], rep)

            save(conf_preds_merged, [self.__output_dir, FileNames.predictions_folder], rep,
                 lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

            save(confs_info, [self.__output_dir, FileNames.configurations_folder, FileNames.configurations_info_folder],
                 rep, lambda o: o.to_dict() if isinstance(o, ModelConf) else o)
        return perfs_dict, predictions_ordered, confs_info

    def __cross_validate_classifiers(self, X, Y, folds_inds, fsels_in_folds, rep):
        conf_info = OrderedDict()
        predictions = OrderedDict()
        elapsed_time = 0.0
        for fold_id, inds in folds_inds.items():
            start = time.time()
            train_inds = inds['train_indices']
            test_inds = inds['test_indices']
            assert not np.array_equal(train_inds, test_inds)
            X_train, X_test = X.iloc[train_inds, :], X.iloc[test_inds, :]
            Y_train, Y_test = Y[train_inds], Y[test_inds]
            total_combs = len(fsels_in_folds[fold_id]) * len(self.__clf_configs)
            conf_id = 0
            for fsel in fsels_in_folds[fold_id]:
                X_train_new = X_train.iloc[:, fsel.get_features()]
                X_test_new = X_test.iloc[:, fsel.get_features()]
                assert 0 < len(fsel.get_features()) == X_test_new.shape[1]
                for clf_conf in self.__clf_configs:
                    predictions.setdefault(conf_id, {})
                    classifier = Classifier(clf_conf)
                    CV_Classification.__console_log(rep, fold_id, conf_id + 1, total_combs, fsel, classifier, elapsed_time)
                    predictions_proba = classifier.train(X_train_new, Y_train).predict_proba(X_test_new)
                    predictions[conf_id][fold_id] = predictions_proba
                    conf_info[conf_id] = ModelConf(fsel, classifier, conf_id)
                    conf_id += 1
            assert len(predictions) == total_combs and len(conf_info) == total_combs
            elapsed_time = time.time() - start
        return conf_info, predictions

    def __train_best_model_in_all_data(self, best_model_per_metric, confs_info, X, Y):
        for m_id, m_data in best_model_per_metric.items():
            for best_c_id in m_data:
                start = time.time()
                conf = confs_info[best_c_id]
                fsel = conf.get_fsel()
                print('\rMetric', m_id, ': Training in all data the config', conf.get_fsel().get_id(), '>',
                      conf.get_clf().get_id(), end='')
                if not self.__is_explanation:
                    fsel = FeatureSelection(conf.get_fsel().get_config())
                    fsel.run(X, Y)
                    if self.__knowledge_discovery is True and automlconsts.SELECT_TOPK_FEATURES:
                        fsel.set_features(fsel.get_features()[0:automlconsts.MAX_FEATURES])
                    assert len(fsel.get_features()) > 0
                    end = time.time()
                    fsel.set_time(round(end - start, 2))
                X_new = X.iloc[:, fsel.get_features()]
                clf = Classifier(conf.get_clf().get_config())
                start = time.time()
                clf.train(X_new, Y)
                end = time.time()
                clf.set_time(round(end - start, 2))
                best_model_per_metric[m_id] = ModelConf(fsel, clf, conf.get_conf_id())
        print()
        return best_model_per_metric

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
            metrics_dict = calculate_all_metrics(true_labels, preds, automlconsts.METRICS_TO_CALCULATE)
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

    @ staticmethod
    def __folds_true_labels(Y, folds_inds):
        folds_true_labels = {}
        true_labels = np.array([])
        for fold_id, inds in folds_inds.items():
            true_labels = np.concatenate((true_labels, Y[inds['test_indices']]))
            folds_true_labels[fold_id] = Y[inds['test_indices']]
        return true_labels, folds_true_labels

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
    def __merge_test_inds(folds_inds):
        inds = np.array([], dtype=int)
        for f_id in folds_inds:
            inds = np.concatenate((inds, folds_inds[f_id]['test_indices']))
        return inds

    @staticmethod
    def __console_log(rep, fold_id, conf_id, total_confs, fsel, classifier, elapsed_time):
        print('\rRepetition', rep+1, '> Fold', fold_id, ':', conf_id, '/', total_confs, fsel.get_id(), fsel.get_params(),
              '>', classifier.get_id(), classifier.get_params(), 'Time for fold', fold_id-1,
              'was', round(elapsed_time, 2), 'secs', end='')