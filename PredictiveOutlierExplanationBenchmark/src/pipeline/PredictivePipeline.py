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
        print('Predictive pipeline\n')
        train_inds, test_inds = self.__train_test_inds()
        dataset_train = self.__train_data(train_inds)
        dataset_test = self.__test_data(test_inds)
        train_data_with_detected_outliers, detectors_info, threshold = detect_outliers(dataset_train)
        # todo The test data should have the labels of the best outlier detector from train data. This is the labels we should see if we can mimic
        test_data_with_detected_outliers, detectors_info, _, = detect_outliers(dataset_test)
        pseudo_samples_array = SettingsConfig.get_pseudo_samples_array()
        datasets_for_cv = {}
        if pseudo_samples_array is not None:
            assert SettingsConfig.is_classification_task(), "Pseudo samples are allowed only in classification task"
            datasets_for_cv.update(utils.add_datasets_with_pseudo_samples(train_data_with_detected_outliers,
                                                                          detectors_info['best'],
                                                                          threshold, pseudo_samples_array))

        print()
        print('Running Dataset:', DatasetConfig.get_dataset_path())
        rw = None

        detectors_info = self.__run_detector_in_hold_out(detectors_info, dataset_test)

        for pseudo_samples, dataset in datasets_for_cv.items():
            Logger.initialize(pseudo_samples)
            rw = ResultsWriter(pseudo_samples, '../results_predictive')
            rw.write_dataset(dataset, 'detected')
            results = Benchmark.run('detected', pseudo_samples, dataset)
            results = self.__test_best_model_in_hold_out(results, dataset_test)
            rw.write_results(results, 'detected')
            del results
            gc.collect()
        rw.write_detector_info_file(detectors_info['info'])
        rw.create_navigator_file()

    def __run_detector_in_hold_out(self, detectors_info, test_data):
        from sklearn.metrics import roc_auc_score
        scores = detectors_info['best'].predict(test_data.get_X())
        perf = roc_auc_score(test_data.get_Y(), scores)
        detectors_info['best'].set_scores_in_hold_out(scores)
        detectors_info['best'].set_hold_out_effectiveness(perf)
        return detectors_info

    def __test_best_model_in_hold_out(self, results, test_data):
        for key, data in results.items():
            for m_id, conf in data['best_model_trained_per_metric'].items():
                fsel = conf.get_fsel()
                clf = conf.get_clf()
                X_new = test_data.get_X().iloc[:, fsel.get_features()]
                predictions = clf.predict_proba(X_new)
                perf = metrics.calculate_metric(test_data.get_Y(), predictions, m_id)
                conf.set_hold_out_effectiveness(perf, m_id)
                results[key][m_id] = conf
        return results

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