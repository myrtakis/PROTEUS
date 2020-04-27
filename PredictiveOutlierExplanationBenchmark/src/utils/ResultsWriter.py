from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileKeys, FileNames
import json
from pathlib import Path
import os


class ResultsWriter:

    __pseudo_samples_dirs_dict = {}

    def __init__(self, results, pseudo_samples, dataset):
        self.__benchmark_dict = self.__prepare_benchmark_dict(results)
        self.__best_model_dict = self.__prepare_best_models_dict(results)
        self.__pseudo_samples = pseudo_samples
        self.__train_test_indices_dict = self.__prepare_train_tets_indices_dict(results)
        self.__pseudo_samples_key = 'pseudo_samples_' + str(pseudo_samples)
        self.__dataset_path = None
        self.__dataset = dataset
        self.__base_dir = None
        self.__final_dir = None
        self.__detector_info_path = None
        self.__generate_dir()

    def write_results(self):
        self.__dataset_path = os.path.join(self.__final_dir, self.__pseudo_samples_key + '_data.csv')
        with open(os.path.join(self.__final_dir, FileNames.best_models_bench_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__benchmark_dict, indent=4, separators=(',', ': '), ensure_ascii=False))
        with open(os.path.join(self.__final_dir, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__best_model_dict, indent=4, separators=(',', ': '), ensure_ascii=False))
        with open(os.path.join(self.__final_dir, FileNames.train_test_indices_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__train_test_indices_dict, indent=4, separators=(',', ': '), ensure_ascii=False))
        self.__dataset.get_df().to_csv(self.__dataset_path, index=False)
        self.__update_pseudo_samples_dir()

    def create_navigator_file(self):
        assert self.__base_dir is not None
        navigator_dict = {
            FileKeys.navigator_conf_path: ConfigMger.get_config_path(),
            FileKeys.navigator_original_dataset_path: DatasetConfig.get_dataset_path(),
            FileKeys.navigator_detector_info_path: self.__detector_info_path,
            FileKeys.navigator_pseudo_samples_key: ResultsWriter.__pseudo_samples_dirs_dict
        }
        with open(os.path.join(self.__base_dir, FileNames.navigator_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(navigator_dict, indent=4, separators=(',', ': '), ensure_ascii=False))
        ResultsWriter.__pseudo_samples_dirs_dict = {}  # Reset pseudo sample info after navigator file is written

    def write_detector_info_file(self, detectors_info):
        self.__detector_info_path = os.path.join(self.__base_dir, FileNames.detector_info_fname)
        det_dict = {}
        for det_id, det_val in detectors_info.items():
            det_dict.update(det_val.to_dict())
        with open(self.__detector_info_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(det_dict, indent=4, separators=(',', ': '), ensure_ascii=False))

    def __prepare_best_models_dict(self, results):
        tmp_dict = {'no_fs': {}, 'fs': {}}
        for k in tmp_dict.keys():
            for m_id, m_data in results[k]['best_model_trained_per_metric'].items():
                tmp_dict[k][m_id] = m_data.to_dict()
        return tmp_dict

    def __prepare_benchmark_dict(self, results):
        tmp_dict = {'no_fs': {}, 'fs': {}}
        for k in tmp_dict.keys():
            for c_id, c_data in results[k]['conf_data_in_folds'].items():
                for f_id, f_data in c_data.items():
                    tmp_dict[k].setdefault(c_id, {})
                    tmp_dict[k][c_id][f_id] = f_data.to_dict()
        return tmp_dict

    def __prepare_train_tets_indices_dict(self, results):
        tmp_dict = {'no_fs': {}, 'fs': {}}
        for k in tmp_dict.keys():
            for f_id, f_data in results[k]['train_test_indices_folds'].items():
                tmp_dict[k].setdefault(f_id, {})
                for ind_k, ind_data in f_data.items():
                    tmp_dict[k][f_id][ind_k] = [int(x) for x in ind_data]
        return tmp_dict

    def __generate_dir(self):
        dataset_path = DatasetConfig.get_dataset_path()
        if dataset_path.startswith('..'):
            dataset_path = os.path.join(*Path(dataset_path).parts[1:])
        outlier_ratio_str = str(SettingsConfig.get_top_k_points_to_explain()).replace('.', '')
        base_name = os.path.splitext(os.path.basename(dataset_path))[0] + '_' + outlier_ratio_str
        dataset_path = dataset_path.replace(os.path.basename(dataset_path), '')
        self.__base_dir = Path(
            FileNames.default_folder,
            SettingsConfig.get_task(),
            dataset_path,
            base_name)
        self.__final_dir = os.path.join(self.__base_dir, self.__pseudo_samples_key)
        Path(self.__final_dir).mkdir(parents=True, exist_ok=True)

    def __update_pseudo_samples_dir(self):
        ResultsWriter.__pseudo_samples_dirs_dict[self.__pseudo_samples_key] = {
            FileKeys.navigator_pseudo_samples_num_key: self.__pseudo_samples,
            FileKeys.navigator_pseudo_sample_dir_key: self.__final_dir,
            FileKeys.navigator_pseudo_samples_data_path: self.__dataset_path
        }

    def get_base_dir(self):
        return self.__base_dir
