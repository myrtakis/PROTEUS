from collections import OrderedDict
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection


class CV_Fselection:

    __SELECT_TOPK_FEATURES = True
    __MAX_FEATURES = 10

    def __init__(self, knowledge_discovery, fsel_configs):
        self.__knowledge_discovery = knowledge_discovery
        self.__fsel_configs = fsel_configs
        self.__valid_fs_confs = np.repeat(True, len(fsel_configs))
        self.__kfolds = None

    def run(self, folds_inds_reps, dataset):
        self.__kfolds = len(folds_inds_reps)
        reps_fsel_fold_array = np.full((len(folds_inds_reps), len(self.__fsel_configs)), np.nan, dtype=object)
        for rep, folds_inds in folds_inds_reps.items():
            print('Repetition', rep, end='')
            fsel_fold_dict = self.__cross_validate_fs(dataset.get_X(), dataset.get_Y(), folds_inds)
            reps_fsel_fold_array[rep, :] = fsel_fold_dict
        valid_confs = self.__keep_valid_confs(reps_fsel_fold_array)
        for rep in range(valid_confs.shape[0]):
            self.__select_top_k_features(valid_confs[rep, :])
        folds_to_confs = OrderedDict()
        for rep in range(valid_confs.shape[0]):
            folds_to_confs[rep] = self.__folds_to_confs_restructure(valid_confs[rep, :])
        return folds_to_confs

    def __cross_validate_fs(self, X, Y, folds_inds):
        fsel_fold_array = np.full(self.__fsel_configs, np.nan, dtype=object)
        for fold_id, inds in folds_inds.items():
            train_inds = inds['train_indices']
            test_inds = inds['test_indices']
            assert not np.array_equal(train_inds, test_inds)
            X_train, X_test = X.iloc[train_inds, :], X.iloc[test_inds, :]
            Y_train, Y_test = Y[train_inds], Y[test_inds]
            conf_id = 0
            for fsel_conf in self.__fsel_configs:
                if CV_Fselection.__omit_fsel(fsel_conf, self.__knowledge_discovery):
                    continue
                if self.__knowledge_discovery is True:
                    print('\r', '>Fold', fold_id, '> Running fsel:', fsel_conf, end='')
                fsel = FeatureSelection(fsel_conf)
                fsel.run(X_train, Y_train)
                print(' Completed', end='')
                if np.isnan(fsel_fold_array[conf_id]):
                    fsel_fold_array[conf_id] = OrderedDict()
                    fsel_fold_array[conf_id][fold_id] = fsel
                if len(fsel.get_features()) == 0:
                    self.__valid_fs_confs[conf_id] = False
                conf_id += 1
        assert not np.isnan(fsel_fold_array).any()
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
                if len(fsel.get_features()) > CV_Fselection.__MAX_FEATURES:
                    fsel.set_features(fsel.get_features()[0:CV_Fselection.__MAX_FEATURES])

    def __folds_to_confs_restructure(self, fsel_confs):
        folds_to_confs_dict = OrderedDict()
        for conf_obj in fsel_confs:
            assert len(conf_obj) == self.__kfolds
            for f_id, fsel in conf_obj.items():
                assert len(fsel.get_features()) <= CV_Fselection.__MAX_FEATURES
                folds_to_confs_dict.setdefault(f_id, [])
                folds_to_confs_dict[f_id].append(fsel)
        return folds_to_confs_dict

    @staticmethod
    def __omit_fsel(fsel_conf, knowledge_discovery):
        if knowledge_discovery is True and 'none' in fsel_conf['id'].lower():
            return True
        if knowledge_discovery is False and 'none' not in fsel_conf['id'].lower():
            return True
        return False
