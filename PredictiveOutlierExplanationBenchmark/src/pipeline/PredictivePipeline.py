import gc
from sklearn.model_selection import StratifiedShuffleSplit
from PredictiveOutlierExplanationBenchmark.src.configpkg import SettingsConfig
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.pipeline.Benchmark import Benchmark, DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.pipeline.Detection import detect_outliers
from PredictiveOutlierExplanationBenchmark.src.utils import utils, metrics
from PredictiveOutlierExplanationBenchmark.src.utils.Logger import Logger
from PredictiveOutlierExplanationBenchmark.src.utils.ResultsWriter import ResultsWriter


class PredictivePipeline:

    __HOLD_OUD_PERCENTAGE = 0.5

    def __init__(self, save_dir, original_dataset):
        self.save_dir = save_dir
        self.original_dataset = original_dataset

    def run(self):
        print('Predictive pipeline pipeline\n')
        train_inds, test_inds = self.__train_test_inds()
        dataset_train = self.__train_data(train_inds)
        dataset_test = self.__test_data(test_inds)
        dataset_with_detected_outliers, detectors_info, threshold = detect_outliers(dataset_train)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        datasets_for_cv = {}
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv['detected'].update(utils.add_datasets_with_pseudo_samples(dataset_with_detected_outliers,
                                                                                      detectors_info['best'],
                                                                                      threshold, pseudo_samples_array))
        # todo the results now should be stored in a folder with name "results_predictive_pipeline"
        print()
        print('Running Dataset:', DatasetConfig.get_dataset_path())
        rw = None

        self.__test_detector_in_hold_out(detectors_info['best'], dataset_test)

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

    def __test_detector_in_hold_out(self, detector, test_data):
        from sklearn.metrics import roc_auc_score
        scores = detector.predict(test_data.get_X())
        perf = roc_auc_score(test_data.get_Y(), scores)
        detector.set_scores_in_hold_out(scores)
        detector.set_hold_out_effectiveness(perf)
        return detector

    def __test_best_model_in_hold_out(self, best_models_per_metric, test_data):
        for m_id, conf in best_models_per_metric.items():
            fsel = conf.get_clf()
            clf = conf.get_clf()
            X_new = test_data.get_X().iloc[:, fsel.get_features()]
            predictions = clf.predict(X_new)
            perf = metrics.calculate_metric(test_data.get_Y(), predictions, m_id)
            conf.set_hold_out_effectiveness(perf, m_id)
            best_models_per_metric[m_id] = conf
        return best_models_per_metric

    def __train_test_inds(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=PredictivePipeline.__HOLD_OUD_PERCENTAGE, random_state=0)
        train_inds, test_inds = next(sss.split(self.original_dataset.get_X(), self.original_dataset.get_Y()))
        return train_inds, test_inds

    def __train_data(self, train_inds):
        return Dataset(self.original_dataset.get_df().iloc[train_inds, :],
                       self.original_dataset.get_anomaly_column_name(),
                       self.original_dataset.get_subspace_column_name())

    def __test_data(self, test_inds):
        return Dataset(self.original_dataset.get_df().iloc[test_inds, :],
                       self.original_dataset.get_anomaly_column_name(),
                       self.original_dataset.get_subspace_column_name())