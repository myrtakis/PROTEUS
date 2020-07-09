import gc
from sklearn.model_selection import StratifiedShuffleSplit
from baselines.posthoc_explanation_methods import ExplanationMethods
from configpkg import SettingsConfig
from holders.Dataset import Dataset
from pipeline.Benchmark import Benchmark, DatasetConfig
from pipeline.Detection import evaluate_detectors, detect
from pipeline.automl.automl_processor import AutoML
from utils import helper_functions, metrics
from utils.ResultsWriter import ResultsWriter
from pathlib import Path
import time


class PredictivePipeline:

    __HOLD_OUD_PERCENTAGE = 0.3

    __methods_internal_oversampling = [
        'micencova'
    ]

    def __init__(self, save_dir, original_dataset, oversampling_method, detector=None):
        self.save_dir = save_dir
        # self.original_dataset = original_dataset
        self.original_dataset = helper_functions.add_noise_to_data(original_dataset)
        if self.original_dataset.get_X().shape[1] > original_dataset.get_X().shape[1]:
            print('Noise added', original_dataset.get_X().shape[1], 'dimensions ->', self.original_dataset.get_X().shape[1])
        self.oversampling_method = oversampling_method
        detector = 'best_detector' if detector is None else detector
        self.protean_results_dir = Path('..', 'results_predictive', detector, 'protean',
                                        oversampling_method + '_oversampling')
        self.baselines_results_dir = Path('..', 'results_predictive', detector, 'baselines')
        self.detector = detector

    def run(self):
        print('Predictive pipeline\n')
        train_inds, test_inds = self.__train_test_inds()
        original_dataset_train = self.__train_data(train_inds)
        original_dataset_test = self.__test_data(test_inds)
        train_data_with_detected_outliers, detectors_info, threshold = evaluate_detectors(original_dataset_train, self.detector)
        detectors_info, test_data_labels = self.__run_detector_in_hold_out(detectors_info, original_dataset_test)
        test_data_with_detected_outliers = self.__generate_test_dataset(test_data_labels, original_dataset_test)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        datasets_for_cv = {}
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv.update(helper_functions.add_datasets_with_pseudo_samples(self.oversampling_method,
                                                                                     train_data_with_detected_outliers,
                                                                                     detectors_info['best'],
                                                                                     threshold, pseudo_samples_array))

        print()
        print('Running Dataset:', DatasetConfig.get_dataset_path())

        explanation_methods = ExplanationMethods(train_data_with_detected_outliers, detectors_info['best'])
        baseline_explanations = explanation_methods.run_all_post_hoc_explanation_methods()

        if detectors_info['best'].get_detector().is_explainable():
            explanation = PredictivePipeline.__detector_explanation(train_data_with_detected_outliers, detectors_info['best'])
            baseline_explanations[detectors_info['best'].get_id()] = explanation

        ResultsWriter.setup_writer(self.protean_results_dir)
        ResultsWriter.write_train_hold_out_inds({'training_inds': train_inds.tolist(), 'holdout_inds': test_inds.tolist()})
        ResultsWriter.write_detector_info_file(detectors_info['best'])
        ResultsWriter.write_baselines_explanations(baseline_explanations, self.baselines_results_dir)

        for pseudo_samples, dataset in datasets_for_cv.items():
            print('--------------\n--------------')
            print('Running dataset with pseudo samples: ', pseudo_samples)
            rw = ResultsWriter(pseudo_samples)
            rw.write_dataset(dataset, 'detected')
            rw.write_hold_out_dataset(test_data_with_detected_outliers)
            PredictivePipeline.__build_write_baselines_models(train_data_with_detected_outliers,
                                                              test_data_with_detected_outliers,
                                                              rw, baseline_explanations, pseudo_samples)
            best_trained_model_nofs = AutoML(rw.get_final_dir()).run(dataset, False)
            best_trained_model_fs = AutoML(rw.get_final_dir()).run(dataset, True)
            results = {'no_fs': best_trained_model_nofs, 'fs': best_trained_model_fs}
            results = PredictivePipeline.__test_best_model_in_hold_out(results, test_data_with_detected_outliers)
            rw.write_results(results, 'detected')
            ResultsWriter.flush_navigator_file()
            del best_trained_model_nofs
            del best_trained_model_fs
            del results
            gc.collect()

    def __generate_test_dataset(self, test_data_labels, original_dataset_test):
        anomaly_column = original_dataset_test.get_anomaly_column_name()
        subspace_column = original_dataset_test.get_subspace_column_name()
        new_df = original_dataset_test.get_df().copy()
        new_df[anomaly_column] = test_data_labels
        return Dataset(new_df, anomaly_column, subspace_column)

    def __run_detector_in_hold_out(self, detectors_info, original_dataset_test):
        test_data_labels, scores, perf = detect(detectors_info['best'], original_dataset_test.get_X(),
                                                original_dataset_test.get_Y())
        detectors_info['best'].set_scores_in_hold_out(scores)
        detectors_info['best'].set_hold_out_effectiveness(perf)
        return detectors_info, test_data_labels

    @staticmethod
    def __test_best_model_in_hold_out(best_model, test_data):
        for key, data in best_model.items():
            for m_id, conf in data.items():
                fsel = conf.get_fsel()
                clf = conf.get_clf()
                X_new = test_data.get_X().iloc[:, fsel.get_features()]
                predictions = clf.predict_proba(X_new)
                perf = metrics.calculate_metric(test_data.get_Y(), predictions, m_id)
                conf.set_hold_out_effectiveness(perf, m_id)
                best_model[key][m_id] = conf
        return best_model

    def __train_test_inds(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=PredictivePipeline.__HOLD_OUD_PERCENTAGE, random_state=0)
        train_inds, test_inds = next(sss.split(self.original_dataset.get_X(), self.original_dataset.get_Y()))
        return train_inds, test_inds

    def __train_data(self, train_inds):
        return Dataset(self.original_dataset.get_df().iloc[train_inds, :],
                       self.original_dataset.get_anomaly_column_name(),
                       self.original_dataset.get_subspace_column_name(), True)

    def __test_data(self, test_inds):
        return Dataset(self.original_dataset.get_df().iloc[test_inds, :],
                       self.original_dataset.get_anomaly_column_name(),
                       self.original_dataset.get_subspace_column_name(), True)

    @staticmethod
    def __detector_explanation(train_data_with_detected_outliers, detector_obj):
        print('Computing the explanation for', detector_obj.get_id())
        start = time.time()
        detector_obj.get_detector(). \
            calculate_explanation(train_data_with_detected_outliers.get_outlier_indices())
        return {
            'time': time.time() - start,
            'global_explanation': detector_obj.get_detector().convert_to_global_explanation(),
            'local_explanation': detector_obj.get_detector().get_explanation()
        }

    @staticmethod
    def __build_write_baselines_models(train_dataset, test_dataset, rw, baselines_methods, pseudo_samples):
        for method in baselines_methods:
            global_expl = baselines_methods[method]['global_explanation']
            if pseudo_samples > 0 and method in PredictivePipeline.__methods_internal_oversampling:
                print('----\nMethod', method, 'does internal oversampling and omitted for the oversampling dataset')
                continue
            else:
                print('----\nBuilding model for method', method)
            best_model = {'fs': AutoML(rw.get_final_baseline_dir(method)).run(train_dataset, False, global_expl)}
            best_model = PredictivePipeline.__test_best_model_in_hold_out(best_model, test_dataset)
            rw.write_baseline_results(best_model, method)
