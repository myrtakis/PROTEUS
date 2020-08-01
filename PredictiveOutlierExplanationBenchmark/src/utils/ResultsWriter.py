from configpkg import *
from utils.shared_names import FileKeys, FileNames
import json
from pathlib import Path
import os
import numpy as np


class ResultsWriter:

    __base_dir = None
    __baselines_dir = None
    __navigator_file_dict = None
    __navigator_file_path = None
    __results_initial_dir = None
    __noise = None

    def __init__(self, pseudo_samples):
        assert ResultsWriter.__base_dir is not None
        assert ResultsWriter.__navigator_file_dict is not None
        self.__pseudo_samples = pseudo_samples
        self.__pseudo_samples_key = 'pseudo_samples_' + str(pseudo_samples)
        self.__final_dir = None
        self.__generate_final_dir()
        self.__update_pseudo_samples_dir(FileKeys.navigator_pseudo_samples_num_key, pseudo_samples)
        self.__update_pseudo_samples_dir(FileKeys.navigator_pseudo_sample_dir_key, self.__final_dir)

    def write_results(self, results, dataset_kind):
        # if dataset_kind == 'original':
        #     original_data_res_path = Path(ResultsWriter.__base_dir, dataset_kind)
        #     original_data_res_path.mkdir(parents=True, exist_ok=True)
        #     output_dir = original_data_res_path
        #     self.__update_and_flush_navigator_file(FileKeys.navigator_original_data_results, original_data_res_path)
        best_models_dict = ResultsWriter.__prepare_results(results)
        for expl_size, best_model in best_models_dict.items():
            output_dir = ResultsWriter.add_explanation_size_to_path(self.__final_dir, expl_size)
            ResultsWriter.write_info_file(output_dir, expl_size)
            with open(os.path.join(output_dir, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
                f.write(json.dumps(best_model, indent=4, separators=(',', ': '), ensure_ascii=False))

    def write_baseline_results(self, best_model_per_expl_size, method_id):
        method_output_dir = self.get_final_baseline_dir(method_id)
        for key, expl_data in best_model_per_expl_size.items():
            best_model_to_write = {'fs': {}}
            for expl_size, best_model in expl_data.items():
                for m_id, m_data in best_model.items():
                    best_model_to_write['fs'][m_id] = m_data.to_dict()
                final_output_path = ResultsWriter.add_noise_to_path(method_output_dir)
                final_output_path = ResultsWriter.add_explanation_size_to_path(final_output_path, expl_size)
                ResultsWriter.write_info_file(final_output_path, expl_size)
                with open(os.path.join(final_output_path, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(best_model_to_write, indent=4, separators=(',', ': '), ensure_ascii=False))
        self.__update_baselines_dict(method_id, {FileKeys.navigator_pseudo_sample_dir_key: method_output_dir,
                                                 FileKeys.navigator_pseudo_samples_num_key: self.__pseudo_samples})

    @staticmethod
    def write_info_file(output_folder, expl_size):
        info_file_cont = {FileKeys.info_file_explanation_size: expl_size,
                          FileKeys.info_file_noise_level: ResultsWriter.__noise}
        with open(os.path.join(output_folder, FileNames.info_file_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(info_file_cont, indent=4, separators=(',', ': '), ensure_ascii=False))

    @staticmethod
    def __prepare_results(results):
        best_model_per_expl_size = {}
        for key, data in results.items():
            for expl_size, best_model in data.items():
                best_model_per_expl_size.setdefault(expl_size, {})
                for m_id, m_data in best_model.items():
                    best_model[m_id] = m_data.to_dict()
                best_model_per_expl_size[expl_size][key] = best_model
        return best_model_per_expl_size

    def write_dataset(self, dataset, dataset_kind):
        if dataset_kind == 'original':
            return
        dataset_path = os.path.join(self.__final_dir, self.__pseudo_samples_key + '_data.csv')
        dataset.get_df().to_csv(dataset_path, index=False)
        self.__update_pseudo_samples_dir(FileKeys.navigator_pseudo_samples_data_path, dataset_path)

        pseudo_samples_info_path = Path(self.__final_dir, FileNames.pseudo_samples_info)
        if dataset.get_pseudo_sample_indices_per_outlier() is not None:
            ps_info = dict((int(o), list(range(ps_range[0], ps_range[1]))) for o, ps_range in
                           dataset.get_pseudo_sample_indices_per_outlier().items())
            with open(pseudo_samples_info_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(ps_info, indent=4, separators=(',', ': '), ensure_ascii=False))
        self.__update_pseudo_samples_dir(FileKeys.navigator_pseudo_samples_inds, pseudo_samples_info_path)

    @staticmethod
    def write_train_hold_out_inds(inds):
        train_hold_out_path = Path(ResultsWriter.__base_dir, FileNames.train_hold_out_indices_fname)
        with open(train_hold_out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(inds, indent=4, separators=(',', ': '), ensure_ascii=False))
        ResultsWriter.__update_and_flush_navigator_file(FileKeys.navigator_train_hold_out_inds, train_hold_out_path )

    def write_hold_out_dataset(self, hold_out_dataset):
        hold_out_dataset_path = Path(self.__final_dir, self.__pseudo_samples_key + '_hold_out_data.csv')
        hold_out_dataset.get_df().to_csv(hold_out_dataset_path, index=False)
        self.__update_pseudo_samples_dir(FileKeys.navigator_pseudo_samples_hold_out_data_key, hold_out_dataset_path)

    @staticmethod
    def write_detector_info_file(best_detector):
        detector_info_path = Path(ResultsWriter.__base_dir, FileNames.detector_info_fname)
        with open(detector_info_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(best_detector.to_dict(), indent=4, separators=(',', ': '), ensure_ascii=False))
        ResultsWriter.__update_and_flush_navigator_file(FileKeys.navigator_detector_info_path,
                                                        detector_info_path)

    @staticmethod
    def write_baselines_explanations(explanations, start_dir):
        ResultsWriter.__baselines_dir = str(ResultsWriter.__base_dir).replace(str(ResultsWriter.__results_initial_dir), str(start_dir))
        Path(ResultsWriter.__baselines_dir).mkdir(parents=True, exist_ok=True)
        baseline_expl_file_path = Path(ResultsWriter.__baselines_dir , FileNames.baselines_explanations_fname)
        with open(baseline_expl_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(explanations, default=ResultsWriter.json_type_transformer,
                               indent=4, separators=(',', ': '), ensure_ascii=False))
        ResultsWriter.__update_and_flush_navigator_file(FileKeys.navigator_baselines_dir_key, ResultsWriter.__baselines_dir)

    @staticmethod
    def setup_writer(results_dir, noise):
        ResultsWriter.__noise = noise
        ResultsWriter.__setup_base_dir(results_dir).mkdir(parents=True, exist_ok=True)
        ResultsWriter.__setup_navigator_file()

    @staticmethod
    def __setup_navigator_file():
        if ResultsWriter.__navigator_file_dict is not None:
            return
        ResultsWriter.__navigator_file_path = Path(ResultsWriter.__base_dir, FileNames.navigator_fname)
        if ResultsWriter.__navigator_file_path.exists() and not ResultsWriter.__nav_file_is_empty():
            with open(ResultsWriter.__navigator_file_path) as json_file:
                ResultsWriter.__navigator_file_dict = json.load(json_file)
            return
        ResultsWriter.__navigator_file_dict = {
            FileKeys.navigator_conf_path: ConfigMger.get_config_path(),
            FileKeys.navigator_original_dataset_path: DatasetConfig.get_dataset_path(),
            FileKeys.navigator_detector_info_path: None,
            FileKeys.navigator_original_data_results: None,
            FileKeys.navigator_train_hold_out_inds: None,
            FileKeys.navigator_pseudo_samples_key: {},
            FileKeys.navigator_baselines_key: {}
        }

    @staticmethod
    def __setup_base_dir(results_dir):
        dataset_path = DatasetConfig.get_dataset_path()
        if dataset_path.startswith('..'):
            dataset_path = os.path.join(*Path(dataset_path).parts[1:])
        outlier_ratio_str = str(SettingsConfig.get_top_k_points_to_explain()).replace('.', '')
        base_name = os.path.splitext(os.path.basename(dataset_path))[0] + '_' + outlier_ratio_str
        dataset_path = dataset_path.replace(os.path.basename(dataset_path), '')
        results_folder = results_dir if results_dir is not None else FileNames.default_folder
        ResultsWriter.__results_initial_dir = results_folder
        ResultsWriter.__base_dir = Path(
            results_folder,
            SettingsConfig.get_task(),
            dataset_path,
            base_name)
        return ResultsWriter.__base_dir

    @staticmethod
    def __update_and_flush_navigator_file(key, val):
        ResultsWriter.__navigator_file_dict[key] = val
        ResultsWriter.flush_navigator_file()

    def __generate_final_dir(self):
        self.__final_dir = os.path.join(ResultsWriter.__base_dir, self.__pseudo_samples_key, ResultsWriter.get_noise_as_name())
        Path(self.__final_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def flush_navigator_file():
        with open(os.path.join(ResultsWriter.__navigator_file_path), 'w', encoding='utf-8') as f:
            f.write(json.dumps(ResultsWriter.__navigator_file_dict, indent=4, separators=(',', ': '),
                               default=ResultsWriter.json_type_transformer, ensure_ascii=False))

    @staticmethod
    def json_type_transformer(o):
        if isinstance(o, Path):
            return str(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return o

    def __update_pseudo_samples_dir(self, key, val):
        if self.__pseudo_samples_key not in ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key]:
            ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key][self.__pseudo_samples_key] = {}
        ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key][self.__pseudo_samples_key][key] = val

    def __update_baselines_dict(self, method_id, val):
        baselines_methods_dict = ResultsWriter.__navigator_file_dict[FileKeys.navigator_baselines_key]
        baselines_methods_dict.setdefault(method_id, {})
        baselines_methods_dict[method_id][self.__pseudo_samples_key] = val
        ResultsWriter.__navigator_file_dict[FileKeys.navigator_baselines_key] = baselines_methods_dict

    @staticmethod
    def __nav_file_is_empty():
        return Path(ResultsWriter.__navigator_file_path).stat().st_size == 0

    @staticmethod
    def get_base_dir():
        assert ResultsWriter.__base_dir is not None
        return ResultsWriter.__base_dir

    @staticmethod
    def get_baselines_dir():
        assert ResultsWriter.__baselines_dir is not None
        return ResultsWriter.__baselines_dir

    def get_final_dir(self):
        return self.__final_dir

    def get_final_baseline_dir(self, method_id):
        method_output_dir = Path(ResultsWriter.__baselines_dir, method_id, self.__pseudo_samples_key)
        method_output_dir.mkdir(parents=True, exist_ok=True)
        return method_output_dir

    @staticmethod
    def add_noise_to_path(path):
        return Path(path, ResultsWriter.get_noise_as_name())

    @staticmethod
    def add_explanation_size_to_path(path, explanation_size):
        return Path(path, 'expl_size_' + str(explanation_size))

    @staticmethod
    def get_noise_as_name():
        return 'noise_' + str(ResultsWriter.__noise).replace('.', '')
