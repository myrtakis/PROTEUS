import gc

from PredictiveOutlierExplanationBenchmark.src.configpkg import SettingsConfig, DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.pipeline.Benchmark import Benchmark
from PredictiveOutlierExplanationBenchmark.src.utils import utils
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import detect_outliers
from PredictiveOutlierExplanationBenchmark.src.utils.Logger import Logger
from PredictiveOutlierExplanationBenchmark.src.utils.ResultsWriter import ResultsWriter


class NormalPipeline:
    def __init__(self, save_dir, original_dataset):
        self.save_dir = save_dir
        self.original_dataset = original_dataset

    def run(self):
        print('Normal pipeline\n')
        datasets_for_cv = {}
        datasets_for_cv['original'] = {0: self.original_dataset}
        datasets_for_cv['detected'] = {}
        dataset_with_detected_outliers, detectors_info, threshold = detect_outliers(self.original_dataset)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv['detected'].update(utils.add_datasets_with_pseudo_samples(dataset_with_detected_outliers,
                                                                                      detectors_info['best'],
                                                                                      threshold, pseudo_samples_array))

        print('Running Dataset:', DatasetConfig.get_dataset_path())
        rw = None

        for dataset_kind, data in datasets_for_cv.items():
            for pseudo_samples, dataset in data.items():
                Logger.initialize(pseudo_samples)
                rw = ResultsWriter(pseudo_samples)
                rw.write_dataset(dataset, dataset_kind)
                results = Benchmark.run(dataset_kind, pseudo_samples, dataset)
                rw.write_results(results, dataset_kind)
                del results
                gc.collect()
        rw.write_detector_info_file(detectors_info['info'])
        rw.create_navigator_file()