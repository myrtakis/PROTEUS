import json
from PredictiveOutlierExplanationBenchmark.src.configpkg.ConfigMger import *
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import *
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import detect_outliers


def run(config_file_path, save_dir):
    config_mger = ConfigMger(config_file_path)
    original_dataset = Dataset(config_mger.get_DatasetConf().get_dataset_path(),
                               config_mger.get_DatasetConf(),
                               config_mger.get_SettingsConf())
    dataset_det_outliers = detect_outliers(config_mger.get_DetectorConf(), original_dataset)


def __cross_validation():
    pass