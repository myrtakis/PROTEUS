from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import SettingsConfig
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
import numpy as np
import pandas as pd


class Transformer:

    def __init__(self):
        pass

    @staticmethod
    def add_pseudo_samples_naive(dataset, pseudo_samples_per_outlier, detector, threshold):
        assert pseudo_samples_per_outlier > 0, "Pseudo samples number should be greater than 0"
        if pseudo_samples_per_outlier == 0:
            return dataset
        total_pseudo_samples = pseudo_samples_per_outlier * len(dataset.get_outlier_indices())
        num_of_original_samples = dataset.get_X().shape[0]
        pseudo_samples_indices = np.arange(num_of_original_samples, num_of_original_samples + total_pseudo_samples)
        new_df = dataset.get_X().copy()
        for outlier in dataset.get_outlier_indices():
            o_sample = new_df.iloc[outlier, :].values
            for ps_sample in range(pseudo_samples_per_outlier):
                pseudo_sample = o_sample + Transformer.__gaussian_noise(dataset)
                new_df = new_df.append(pd.Series(pseudo_sample, index=new_df.columns), ignore_index=True)
        pseudo_samples_df = new_df.iloc[pseudo_samples_indices, :]
        assert pseudo_samples_df.shape[0] == total_pseudo_samples
        pseudo_samples_scores = detector.predict(pseudo_samples_df)
        pseudo_samples_labels = np.zeros(len(pseudo_samples_scores))
        pseudo_samples_labels[np.where(np.array(pseudo_samples_scores) > threshold)] = 1
        new_dataset = Transformer.__dataset_with_anomaly_column(dataset, new_df, pseudo_samples_labels, pseudo_samples_indices)
        return new_dataset

    @staticmethod
    def __gaussian_noise(dataset):
        noise = np.zeros(dataset.get_X().shape[1])
        for i, (mu, sigma) in enumerate(zip(dataset.get_mean_per_column(), dataset.get_std_per_column())):
            noise[i] = np.random.normal(mu, sigma)
        return noise

    @staticmethod
    def __dataset_with_anomaly_column(dataset, new_df, new_anomaly_data, pseudo_samples_indices):
        anomaly_column_name = dataset.get_anomaly_column_name()
        anomaly_data = np.concatenate((dataset.get_Y(), new_anomaly_data), axis=0)
        new_df[anomaly_column_name] = anomaly_data
        subspace_data = dataset.get_subspace_column_data()
        if subspace_data is not None:
            subspace_data = np.concatenate((subspace_data, np.repeat('-', len(pseudo_samples_indices))), axis=0)
            new_df[dataset.get_subspace_column_name()] = subspace_data
        new_dataset = Dataset(new_df, anomaly_column_name, dataset.get_subspace_column_name())
        new_dataset.set_pseudo_samples_indices(pseudo_samples_indices)
        return new_dataset
