from sklearn.model_selection import StratifiedShuffleSplit
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset


class PredictivePipeline:

    __HOLD_OUD_PERCENTAGE = 0.5

    def __init__(self, save_dir, original_dataset):
        self.save_dir = save_dir
        self.original_dataset = original_dataset

    def run(self):
        print('Predictive pipeline pipeline\n')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=PredictivePipeline.__HOLD_OUD_PERCENTAGE, random_state=0)
        train_inds, test_inds = next(sss.split(self.original_dataset.get_X(), self.original_dataset.get_Y()))
        dataset_train = Dataset(self.original_dataset.get_df().iloc[train_inds, :],
                                self.original_dataset.get_anomaly_column_name(),
                                self.original_dataset.get_subspace_column_name())
        dataset_test = Dataset(self.original_dataset.get_df().iloc[test_inds, :],
                                self.original_dataset.get_anomaly_column_name(),
                                self.original_dataset.get_subspace_column_name())
        # todo the results now should be stored in a folder with name "results_predictive_pipeline"
        print()