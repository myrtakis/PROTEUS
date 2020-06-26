import gc
from pathlib import Path

from PredictiveOutlierExplanationBenchmark.src.configpkg import SettingsConfig, DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.pipeline.Benchmark import Benchmark
from PredictiveOutlierExplanationBenchmark.src.pipeline.automl.automl_processor import AutoML
from PredictiveOutlierExplanationBenchmark.src.utils import helper_functions
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import evaluate_detectors
from PredictiveOutlierExplanationBenchmark.src.utils.Logger import Logger
from PredictiveOutlierExplanationBenchmark.src.utils.ResultsWriter import ResultsWriter


class NormalPipeline:

    __RUN_ORIGINAL = False

    def __init__(self, save_dir, original_dataset, oversampling_method, detector=None):
        self.save_dir = save_dir
        self.original_dataset = original_dataset
        self.oversampling_method = oversampling_method
        if detector is None:
            self.results_dir = Path('..', 'results_normal', oversampling_method + '_oversampling', 'best')
        else:
            self.results_dir = Path('..', 'results_normal', oversampling_method + '_oversampling', detector)
        self.detector = detector

    def run(self):
        print('Normal pipeline\n')
        datasets_for_cv = {}
        if NormalPipeline.__RUN_ORIGINAL:
            datasets_for_cv['original'] = {0: self.original_dataset}
        datasets_for_cv['detected'] = {}
        dataset_with_detected_outliers, detectors_info, threshold = evaluate_detectors(self.original_dataset, self.detector)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv['detected'].update(helper_functions.add_datasets_with_pseudo_samples(self.oversampling_method,
                                                                                                 dataset_with_detected_outliers,
                                                                                                 detectors_info['best'],
                                                                                                 threshold, pseudo_samples_array))

        print('Running Dataset:', DatasetConfig.get_dataset_path())
        rw = None

        for dataset_kind, data in datasets_for_cv.items():
            for pseudo_samples, dataset in data.items():
                Logger.initialize(pseudo_samples)
                rw = ResultsWriter(pseudo_samples, self.results_dir)
                rw.write_dataset(dataset, dataset_kind)
                # results = Benchmark.run(dataset_kind, pseudo_samples, dataset, rw.get_final_dir())
                results = AutoML().run(dataset, rw.get_final_dir())
                rw.write_results(results, dataset_kind)
                del results
                gc.collect()
        rw.write_detector_info_file(detectors_info['info'])
        rw.create_navigator_file()
