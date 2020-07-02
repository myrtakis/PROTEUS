from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileKeys, FileNames
import json
from pathlib import Path
import os


class ResultsWriter:

    __base_dir = None
    __navigator_file_dict = None
    __navigator_file_path = None
    __results_initial_dir = None

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
        if dataset_kind == 'original':
            original_data_res_path = Path(ResultsWriter.__base_dir, dataset_kind)
            original_data_res_path.mkdir(parents=True, exist_ok=True)
            output_dir = original_data_res_path
            self.__update_and_flush_navigator_file(FileKeys.navigator_original_data_results, original_data_res_path)
        else:
            output_dir = self.__final_dir
        best_models_dict = ResultsWriter.__prepare_results(results)
        with open(os.path.join(output_dir, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(best_models_dict, indent=4, separators=(',', ': '), ensure_ascii=False))

    @staticmethod
    def __prepare_results(results):
        tmp_dict = {'no_fs': {}, 'fs': {}}
        for k in tmp_dict.keys():
            for m_id, m_data in results[k].items():
                tmp_dict[k][m_id] = m_data.to_dict()
        return tmp_dict

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
    def write_baselines(explanations, start_dir):
        baseline_dir = str(ResultsWriter.__base_dir).replace(str(ResultsWriter.__results_initial_dir), str(start_dir))
        Path(baseline_dir).mkdir(parents=True, exist_ok=True)
        baseline_file_path = Path(baseline_dir, FileNames.baselines_explanations_fname)
        with open(baseline_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(explanations, indent=4, separators=(',', ': '), ensure_ascii=False))
        ResultsWriter.__update_and_flush_navigator_file(FileKeys.navigator_baselines_dir_key, baseline_dir)

    @staticmethod
    def setup_writer(results_dir):
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
            FileKeys.navigator_pseudo_samples_key: {}
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
        self.__final_dir = os.path.join(ResultsWriter.__base_dir, self.__pseudo_samples_key)
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
        else:
            return o

    def __update_pseudo_samples_dir(self, key, val):
        if self.__pseudo_samples_key not in ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key]:
            ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key][self.__pseudo_samples_key] = {}
        ResultsWriter.__navigator_file_dict[FileKeys.navigator_pseudo_samples_key][self.__pseudo_samples_key][key] = val

    @staticmethod
    def __nav_file_is_empty():
        return Path(ResultsWriter.__navigator_file_path).stat().st_size == 0

    @staticmethod
    def get_base_dir():
        assert ResultsWriter.__base_dir is not None
        return ResultsWriter.__base_dir

    def get_final_dir(self):
        return self.__final_dir
