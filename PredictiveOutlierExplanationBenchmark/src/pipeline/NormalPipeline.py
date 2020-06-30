import gc
from pathlib import Path

from PredictiveOutlierExplanationBenchmark.src.baselines.posthoc_explanation_methods import ExplanationMethods
from PredictiveOutlierExplanationBenchmark.src.configpkg import SettingsConfig, DatasetConfig
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
        detector = 'best_detector' if detector is None else detector
        self.protean_results_dir = Path('..', 'results_normal', detector, 'protean',
                                        oversampling_method + '_oversampling')
        self.baselines_results_dir = Path('..', 'results_normal', detector, 'baselines')
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
        if detectors_info['best'].get_detector().is_explainable():
            print('Computing the explanation for', detectors_info['best'].get_id())
            detectors_info['best'].get_detector().\
                calculate_explanation(dataset_with_detected_outliers.get_outlier_indices())

        explanation_methods = ExplanationMethods(dataset_with_detected_outliers, detectors_info['best'])
        explanations = explanation_methods.run_all_post_hoc_explanation_methods()

        ResultsWriter.setup_writer(self.protean_results_dir)
        ResultsWriter.write_detector_info_file(detectors_info['best'])
        ResultsWriter.write_baselines(explanations, self.baselines_results_dir)


        exit()


        for dataset_kind, data in datasets_for_cv.items():
            for pseudo_samples, dataset in data.items():
                Logger.initialize(pseudo_samples)
                rw = ResultsWriter(pseudo_samples)
                rw.write_dataset(dataset, dataset_kind)
                print('----------\nRunning dataset with pseudo samples: ', pseudo_samples)
                best_trained_model_nofs = AutoML().run(dataset, rw.get_final_dir(), False)
                best_trained_model_fs = AutoML().run(dataset, rw.get_final_dir(), True)
                results = {'no_fs': best_trained_model_nofs, 'fs': best_trained_model_fs}
                rw.write_results(results, dataset_kind)
                ResultsWriter.flush_navigator_file()
                del best_trained_model_nofs
                del best_trained_model_fs
                gc.collect()

