from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileKeys, FileNames
import json
from pathlib import Path
import os


class ResultsWriter:

    __pseudo_samples_dirs_dict = {}

    def __init__(self, benchmark_dict, best_model_dict, train_test_indices_dict, pseudo_samples, dataset):
        self.__benchmark_dict = benchmark_dict
        self.__best_model_dict = best_model_dict
        self.__pseudo_samples = pseudo_samples
        self.__train_test_indices_dict = train_test_indices_dict
        self.__pseudo_samples_key = 'pseudo_samples_' + str(pseudo_samples)
        self.__dataset_path = None
        self.__dataset = dataset
        self.__base_dir = None
        self.__final_dir = None
        self.__detector_info_path = None
        self.__generate_dir()

    def write_results(self):
        self.__prepare_benchmark_and_dict()
        self.__prepare_best_model()
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

    def write_detector_info_file(self, detector):
        self.__detector_info_path = os.path.join(self.__base_dir, FileNames.detector_info_fname)
        with open(self.__detector_info_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(detector.to_dict(), indent=4, separators=(',', ': '), ensure_ascii=False))

    def __prepare_best_model(self):
        tmp_dict = {}
        for m_id, metric_data in self.__best_model_dict.items():
            for fsel_clf_id, conf_data in metric_data.items():
                tmp_dict.setdefault(m_id, {})
                tmp_dict[m_id].setdefault(fsel_clf_id, {})
                fsel_dict = conf_data.get_fsel().to_dict()
                clf_dict = conf_data.get_clf().to_dict()
                tmp_dict[m_id][fsel_clf_id].update(
                    **{'effectiveness': conf_data.get_effectiveness()},
                    **{FeatureSelectionConfig.feature_selection_key(): fsel_dict},
                    **{ClassifiersConfig.classifier_key(): clf_dict}
                )
        self.__best_model_dict = tmp_dict

    def __prepare_benchmark_and_dict(self):
        tmp_bench_dict = {}
        for rep, rep_data in self.__benchmark_dict.items():
            tmp_bench_dict.setdefault(rep, {})
            for m_id, metric_data in rep_data.items():
                tmp_bench_dict[rep].setdefault(m_id, {})
                for fsel_clf_id, conf_data in metric_data.items():
                    tmp_bench_dict[rep][m_id].setdefault(fsel_clf_id, {})
                    tmp_bench_dict[rep][m_id][fsel_clf_id] = conf_data.to_dict()
        self.__benchmark_dict = tmp_bench_dict

    def __generate_dir(self):
        dataset_path = DatasetConfig.get_dataset_path()
        if dataset_path.startswith('..'):
            dataset_path = os.path.join(*Path(dataset_path).parts[1:])
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
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
        }

    def get_base_dir(self):
        return self.__base_dir
