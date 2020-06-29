from collections import OrderedDict
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
import PredictiveOutlierExplanationBenchmark.src.pipeline.automl.automl_constants as automlconsts


class CV_Fselection:

    def __init__(self, knowledge_discovery, fsel_configs, kfolds):
        self.__knowledge_discovery = knowledge_discovery
        self.__fsel_configs = CV_Fselection.__filter_fsel_confs(fsel_configs, knowledge_discovery)
        self.__valid_fs_confs = np.repeat(True, len(self.__fsel_configs))
        self.__kfolds = kfolds

    def run(self, folds_inds_reps, dataset):
        print('Knowledge Discovery:', self.__knowledge_discovery)
        reps_fsel_fold_array = np.full((len(folds_inds_reps), len(self.__fsel_configs)), None, dtype=object)
        for rep, folds_inds in folds_inds_reps.items():
            fsel_fold_dict = self.__cross_validate_fs(dataset.get_X(), dataset.get_Y(), folds_inds, rep)
            reps_fsel_fold_array[rep, :] = fsel_fold_dict
        valid_confs = self.__keep_valid_confs(reps_fsel_fold_array)
        for rep in range(valid_confs.shape[0]):
            self.__select_top_k_features(valid_confs[rep, :])
        folds_to_confs = OrderedDict()
        for rep in range(valid_confs.shape[0]):
            folds_to_confs[rep] = self.__folds_to_confs_restructure(valid_confs[rep, :])
        print()
        return folds_to_confs

    def __cross_validate_fs(self, X, Y, folds_inds, rep):
        fsel_fold_array = np.full(len(self.__fsel_configs), None, dtype=object)
        for fold_id, inds in folds_inds.items():
            train_inds = inds['train_indices']
            test_inds = inds['test_indices']
            assert not np.array_equal(train_inds, test_inds)
            X_train, X_test = X.iloc[train_inds, :], X.iloc[test_inds, :]
            Y_train, Y_test = Y[train_inds], Y[test_inds]
            conf_id = 0
            for fsel_conf in self.__fsel_configs:
                if self.__knowledge_discovery is True:
                    print('\rRepetition', rep+1, '> Fold', fold_id, '> Running fsel:', fsel_conf, end='')
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                if fsel_fold_array[conf_id] is None:
                    fsel_fold_array[conf_id] = OrderedDict()
                if len(fsel.get_features()) == 0:
                    self.__valid_fs_confs[conf_id] = False
                fsel_fold_array[conf_id][fold_id] = fsel
                conf_id += 1
        assert None not in fsel_fold_array
        return fsel_fold_array

    def __keep_valid_confs(self, reps_fsel_folds_array):
        if not self.__knowledge_discovery:
            return reps_fsel_folds_array
        valid_confs = np.where(self.__valid_fs_confs == True)[0]
        return reps_fsel_folds_array[:, valid_confs]

    def __select_top_k_features(self, fsel_in_folds):
        if not self.__knowledge_discovery:
            return fsel_in_folds
        for conf_obj in fsel_in_folds:
            for f_id, fsel in conf_obj.items():
                if len(fsel.get_features()) > automlconsts.MAX_FEATURES:
                    fsel.set_features(fsel.get_features()[0:automlconsts.MAX_FEATURES])

    def __folds_to_confs_restructure(self, fsel_confs):
        folds_to_confs_dict = OrderedDict()
        for conf_obj in fsel_confs:
            assert len(conf_obj) == self.__kfolds
            for f_id, fsel in conf_obj.items():
                if self.__knowledge_discovery:
                    assert len(fsel.get_features()) <= automlconsts.MAX_FEATURES
                folds_to_confs_dict.setdefault(f_id, [])
                folds_to_confs_dict[f_id].append(fsel)
        return folds_to_confs_dict

    @staticmethod
    def __filter_fsel_confs(fsel_confs, knowledge_discovery):
        filtered_fsel_confs = []
        for fsel_conf in fsel_confs:
            if knowledge_discovery is True and 'none' not in fsel_conf['id'].lower():
                filtered_fsel_confs.append(fsel_conf)
            if knowledge_discovery is False and 'none' in fsel_conf['id'].lower():
                filtered_fsel_confs.append(fsel_conf)
        return filtered_fsel_confs

