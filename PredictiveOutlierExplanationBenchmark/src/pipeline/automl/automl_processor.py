import collections
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from PredictiveOutlierExplanationBenchmark.src.configpkg import SettingsConfig, ConfigMger, DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.pipeline.automl.cv_feature_selection import CV_Fselection
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileNames
from PredictiveOutlierExplanationBenchmark.src.pipeline.ModelConfigsGen import generate_param_combs
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection



class AutoML:

    __REPETITIONS = 2

    def __init__(self):
        self.__output_dir = None

    def run_with_explanation(self, config_file_path, folds_inds, dataset_path, explanation):
        fsel = FeatureSelection({'id': 'explanation', 'params': None})
        fsel.set_features(explanation)
        ConfigMger.setup_configs(config_file_path)
        dataset = Dataset(dataset_path, DatasetConfig.get_anomaly_column_name(),
                          DatasetConfig.get_subspace_column_name())
        _, classifiers_conf_combs = generate_param_combs()

    def run(self, dataset, output_dir):
        self.__output_dir = output_dir
        kfolds = min(SettingsConfig.get_kfolds(), AutoML.__get_rarest_class_count(dataset))
        assert kfolds > 1, kfolds
        reps_fold_inds, reps_test_inds_merged = self.__create_folds_in_reps(kfolds, dataset, True)
        fsel_conf_combs, classifiers_conf_combs = generate_param_combs()
        no_fs = CV_Fselection(knowledge_discovery=False, fsel_configs=fsel_conf_combs)
        fs = CV_Fselection(knowledge_discovery=True, fsel_configs=fsel_conf_combs)
        print()


    @staticmethod
    def __get_rarest_class_count(dataset):
        if not dataset.contains_pseudo_samples():
            assert dataset.get_pseudo_sample_indices_per_outlier() is None
            return int(min(collections.Counter(dataset.get_Y()).values()))
        else:
            assert dataset.last_original_sample_index() is not None
            assert dataset.get_pseudo_sample_indices_per_outlier() is not None
            return int(min(collections.Counter(dataset.get_Y()[:dataset.last_original_sample_index()]).values()))

    def __create_folds_in_reps(self, kfolds, dataset, save):
        reps_folds_inds = OrderedDict()
        reps_test_inds_merged = OrderedDict()
        for r in range(AutoML.__REPETITIONS):
            skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
            folds_inds = AutoML.__indices_in_folds(dataset, skf)
            reps_folds_inds[r] = folds_inds
            reps_test_inds_merged[r] = AutoML.__merge_inds(folds_inds)
            if save:
                AutoML.__save(folds_inds, [self.__output_dir, FileNames.indices_folder], r,
                              lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        return reps_folds_inds

    @ staticmethod
    def __indices_in_folds(dataset, skf):
        folds_inds = collections.OrderedDict()
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
                train_inds = AutoML.__add_pseudo_samples_inds(ps_indices_per_outlier, train_inds)
                test_inds = AutoML.__add_pseudo_samples_inds(ps_indices_per_outlier, test_inds)
            folds_inds[fold_id] = {'train_indices': train_inds, 'test_indices': test_inds}
            fold_id += 1
        return folds_inds

    @staticmethod
    def __merge_inds(folds_inds):
        inds = np.array([], dtype=int)
        for f_id in folds_inds:
            inds = np.concatenate((inds, folds_inds[f_id]['test_indices']))
        return inds

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

    @staticmethod
    def __save(data, folders_array, repetition, func=lambda o: o):
        path_to_folder = Path(*folders_array)
        path_to_folder.mkdir(parents=True, exist_ok=True)
        output_file = Path(path_to_folder, 'repetition' + str(repetition) + '.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, default=func, indent=4, separators=(',', ': '), ensure_ascii=False))