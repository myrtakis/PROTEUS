from configpkg.SettingsConfig import SettingsConfig
from holders.Dataset import Dataset
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Transformer:

    def __init__(self, method):
        self.method = method

    def transform(self, dataset, pseudo_samples_per_outlier, detector, threshold):
        if self.method == 'random':
            return Transformer.__add_pseudo_samples_random(dataset, pseudo_samples_per_outlier, detector, threshold)
        elif self.method == 'explore':
            return Transformer.__add_pseudo_samples_explore(dataset, pseudo_samples_per_outlier, detector, threshold)
        else:
            assert False, 'Method ' + self.method + ' is unknown'

    @staticmethod
    def __add_pseudo_samples_explore(dataset, pseudo_samples_per_outlier, detector, threshold):
        new_df = dataset.get_X().copy()
        indexes = new_df.index
        inlier_inds = Transformer.__get_curr_inlier_inds(dataset.get_inlier_indices(), indexes)
        new_df = new_df.reset_index(drop=True)
        pseudo_sample_labels_global = []
        for outlier in dataset.get_outlier_indices():
            # print('\rAdding', pseudo_samples_per_outlier, 'pseudo samples for outlier', outlier, end='')
            outlier_row_pos = np.where(indexes == outlier)[0]
            o_sample = new_df.iloc[outlier_row_pos, :]
            closest_inlier = Transformer.nearest_inlier_of(outlier_row_pos, inlier_inds, new_df)
            start_point = o_sample
            end_point = new_df.iloc[closest_inlier, :]
            pseudo_samples = []
            pseudo_samples_labels = []
            while len(pseudo_samples) <= pseudo_samples_per_outlier:
                new_point = (start_point + end_point) / 2
                label = Transformer.label_of_new_point(new_point, detector, threshold)
                pseudo_samples_labels.append(label)
                pseudo_samples.append(new_point)
                if label == 1:
                    start_point = new_point
                else:
                    end_point = new_point
            pseudo_sample_labels_global.extend(pseudo_samples_labels)
            new_df = new_df.append(pseudo_samples, ignore_index=True)
        pseudo_samples_indices = np.arange(dataset.get_X().shape[0], new_df.shape[0])
        new_dataset = Transformer.__dataset_with_anomaly_column(dataset, new_df, pseudo_sample_labels_global,
                                                                pseudo_samples_indices)
        # print()
        return new_dataset

    @staticmethod
    def label_of_new_point(point, detector, threshold):
        new_point_score = detector.predict(point) * -1
        return int(new_point_score > threshold)

    @staticmethod
    def nearest_inlier_of(outlier, inliers_ind, points):
        merged = np.append(inliers_ind, outlier)
        points = points.iloc[merged, :]
        true_points_inds = points.index
        knn = NearestNeighbors(n_neighbors=2).fit(points)
        distances, indices = knn.kneighbors(points.values)
        nneighbor = indices[-1, 1] if indices[-1, 1] != len(merged) else indices[-1, 0]
        return true_points_inds[nneighbor]

    @staticmethod
    def __add_pseudo_samples_random(dataset, pseudo_samples_per_outlier, detector, threshold):
        assert pseudo_samples_per_outlier >= 0, "Pseudo samples number should be greater than 0"
        if pseudo_samples_per_outlier == 0:
            return dataset
        total_pseudo_samples = pseudo_samples_per_outlier * len(dataset.get_outlier_indices())
        num_of_original_samples = dataset.get_X().shape[0]
        pseudo_samples_indices = np.arange(num_of_original_samples, num_of_original_samples + total_pseudo_samples)
        new_df = dataset.get_X().copy()
        indexes = new_df.index
        new_df = new_df.reset_index(drop=True)
        pseudo_samples_inds_per_outlier = {}
        for outlier in dataset.get_outlier_indices():
            outlier_row_pos = np.where(indexes == outlier)[0][0]
            o_sample = new_df.iloc[outlier_row_pos, :].values
            start_ps_ind = new_df.shape[0]
            # print(list(o_sample))
            for ps_sample in range(pseudo_samples_per_outlier):
                pseudo_sample = o_sample + Transformer.__gaussian_noise(dataset, o_sample)
                # print('ps', list(pseudo_sample))
                new_df = new_df.append(pd.Series(pseudo_sample[0], index=new_df.columns), ignore_index=True)
            pseudo_samples_inds_per_outlier[outlier_row_pos] = (start_ps_ind, new_df.shape[0])
        pseudo_samples_df = new_df.iloc[pseudo_samples_indices, :]
        assert pseudo_samples_df.shape[0] == total_pseudo_samples
        pseudo_samples_scores = detector.predict(pseudo_samples_df)
        pseudo_samples_labels = np.zeros(len(pseudo_samples_scores))
        pseudo_samples_labels[np.where(np.array(pseudo_samples_scores) > threshold)] = 1
        new_dataset = Transformer.__dataset_with_anomaly_column(dataset, new_df, pseudo_samples_labels, pseudo_samples_inds_per_outlier)
        return new_dataset

    @staticmethod
    def __gaussian_noise(dataset, o_sample):    # o_sample to be potentially used a the center of the distribution
        noise = np.zeros(dataset.get_X().shape[1])
        for i, (mu, sigma) in enumerate(zip(noise, 0.1 * dataset.get_std_per_column())):
            noise[i] = np.random.normal(mu, sigma)
        return noise

    @staticmethod
    def __get_curr_inlier_inds(true_inlier_inds, curr_inds):
        inlier_inds = []
        for true_inlier_ind in true_inlier_inds:
            inlier_inds.append(np.where(curr_inds == true_inlier_ind)[0])
        return inlier_inds

    @staticmethod
    def __dataset_with_anomaly_column(dataset, new_df, new_anomaly_data, pseudo_samples_indices_per_outlier):
        anomaly_column_name = dataset.get_anomaly_column_name()
        anomaly_data = np.concatenate((dataset.get_Y(), new_anomaly_data), axis=0)
        new_df[anomaly_column_name] = anomaly_data
        subspace_data = dataset.get_subspace_column_data()
        if subspace_data is not None:
            subspace_data = np.concatenate((subspace_data, np.repeat('-', len(new_anomaly_data))), axis=0)
            new_df[dataset.get_subspace_column_name()] = subspace_data
        new_dataset = Dataset(new_df, anomaly_column_name, dataset.get_subspace_column_name())
        new_dataset.set_pseudo_samples_indices_per_outlier(pseudo_samples_indices_per_outlier)
        return new_dataset
