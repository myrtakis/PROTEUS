from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import *
from PredictiveOutlierExplanationBenchmark.src.utils.ResultsWriter import ResultsWriter
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import detect_outliers
from PredictiveOutlierExplanationBenchmark.src.pipeline.Benchmark import Benchmark
from PredictiveOutlierExplanationBenchmark.src.pipeline.DatasetTransformer import Transfomer


class Pipeline:

    def __init__(self):
        pass

    @staticmethod
    def run(config_file_path, save_dir):
        ConfigMger.setup_configs(config_file_path)
        original_dataset = Dataset(DatasetConfig.get_dataset_path(), DatasetConfig.get_anomaly_column_name(),
                                   DatasetConfig.get_subspace_column_name())
        datasets_for_cv = {}
        dataset_with_detected_outliers, detector, threshold = detect_outliers(original_dataset)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv.update(Pipeline.__add_datasets_with_pseudo_samples(dataset_with_detected_outliers, detector,
                                                                               threshold, pseudo_samples_array))

        # todo delete this line (debugging)
        datasets_for_cv[0] = original_dataset

        print('Running Dataset:', DatasetConfig.get_dataset_path())
        rw = None
        for pseudo_samples, dataset in datasets_for_cv.items():
            benchmark_dict, best_model_dict, train_test_indices_dict = Benchmark.run(pseudo_samples, dataset)
            rw = ResultsWriter(benchmark_dict, best_model_dict, train_test_indices_dict, pseudo_samples, dataset)
            rw.write_results()
        rw.write_detector_info_file(detector)
        rw.create_navigator_file()


    # Util Functions

    @staticmethod
    def __add_datasets_with_pseudo_samples(dataset_detected_outliers, detector, threshold, pseudo_samples_arr):
        datasets_with_pseudo_samples = {}
        for ps_num in pseudo_samples_arr:
            if ps_num == 0:
                datasets_with_pseudo_samples[0] = dataset_detected_outliers
                continue
            dataset = Transfomer.add_pseudo_samples_naive(dataset_detected_outliers, ps_num, detector, threshold)
            datasets_with_pseudo_samples[ps_num] = dataset
        return datasets_with_pseudo_samples
