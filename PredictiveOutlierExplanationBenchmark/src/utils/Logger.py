import os
from pathlib import Path

from PredictiveOutlierExplanationBenchmark.src.configpkg import DatasetConfig, SettingsConfig
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileNames


class Logger:

    __logger_name = 'logger.txt'
    __logger_path = None
    __logger_final_file = None

    __make_log = False

    @ staticmethod
    def initialize(pseudo_samples):
        if Logger.__make_log is False:
            return
        if Logger.__logger_final_file is not None:
            Logger.__logger_final_file.close()
        dataset_path = DatasetConfig.get_dataset_path()
        if dataset_path.startswith('..'):
            dataset_path = os.path.join(*Path(dataset_path).parts[1:])
        outlier_ratio_str = str(SettingsConfig.get_top_k_points_to_explain()).replace('.', '')
        base_name = os.path.splitext(os.path.basename(dataset_path))[0] + '_' + outlier_ratio_str
        dataset_path = dataset_path.replace(os.path.basename(dataset_path), '')
        pseudo_samples_key = 'pseudo_samples_' + str(pseudo_samples)
        Logger.__logger_path = Path(
                                    FileNames.default_folder,
                                    SettingsConfig.get_task(),
                                    dataset_path,
                                    base_name,
                                    pseudo_samples_key
        )
        Logger.__logger_path.mkdir(parents=True, exist_ok=True)
        Logger.__logger_final_file = open(Path(Logger.__logger_path, Logger.__logger_name), 'w')

    @ staticmethod
    def log(data):
        if Logger.__make_log is True:
            Logger.__logger_final_file.write(str(data) + '\n')

    @ staticmethod
    def close():
        if Logger.__make_log is True:
            Logger.__logger_final_file.close()
