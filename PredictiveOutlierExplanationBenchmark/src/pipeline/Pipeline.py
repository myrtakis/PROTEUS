import json
from PredictiveOutlierExplanationBenchmark.src.configpkg.ConfigMger import *
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import *
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import detect_outliers
from PredictiveOutlierExplanationBenchmark.src.pipeline.CrossValidation import CrossValidation
from PredictiveOutlierExplanationBenchmark.src.pipeline.DatasetTransformer import Transfomer


class Pipeline:

    def __init__(self):
        pass

    @staticmethod
    def run(config_file_path, save_dir):
        config_mger = ConfigMger.setup_configs(config_file_path)
        original_dataset = Dataset(DatasetConfig.get_dataset_path(), DatasetConfig.get_anomaly_column_name(),
                                   DatasetConfig.get_subspace_column_name())
        datasets_for_cv = {}
        dataset_detected_outliers, detector, threshold = detect_outliers(original_dataset)
        datasets_for_cv['original'] = dataset_detected_outliers
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv.update(Pipeline.__add_datasets_with_pseudo_samples(dataset_detected_outliers, detector,
                                                                               threshold, pseudo_samples_array))
        for dataset_key, dataset in datasets_for_cv.items():
            CrossValidation.run(dataset)

    # Util Functions

    @staticmethod
    def __add_datasets_with_pseudo_samples(dataset_detected_outliers, detector, threshold, pseudo_samples_arr):
        datasets_with_pseudo_samples = {}
        for ps_num in pseudo_samples_arr:
            if ps_num == 0:
                continue
            dataset = Transfomer.add_pseudo_samples_naive(dataset_detected_outliers, ps_num, detector, threshold)
            datasets_with_pseudo_samples[str(ps_num) + '_pseudo_samples'] = dataset
        return datasets_with_pseudo_samples
