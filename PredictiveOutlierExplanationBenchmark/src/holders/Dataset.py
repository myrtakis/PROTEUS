import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, dataset, dataset_conf, settings_conf):
        self.__anomaly_column = dataset_conf.get_anomaly_column()
        self.__subspace_column = dataset_conf.get_subspace_column()
        self.__df_data = None
        self.__labels = None
        self.__scores = None
        self.__subspaces = None
        self.__outlier_indices = None
        self.__inlier_indices = None
        if settings_conf.task_is_classification():
            self.__setup_dataset_for_classification(dataset)
        else:
            self.__setup_dataset_for_regression(dataset)

    def __setup_dataset_for_classification(self, dataset):
        self.__set_df(dataset)
        self.__set_df_data()
        self.__set_labels()
        self.__set_subspaces()
        self.__set_outlier_indices()
        self.__set_inlier_indices()

    def __setup_dataset_for_regression(self, dataset):
        self.__set_df(dataset)
        self.__set_df_data()
        self.__set_scores()

    # Setter Functions

    def __set_df(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            self.__df = dataset
        else:
            self.__df = pd.read_csv(dataset)

    def __set_df_data(self):
        self.__df_data = self.__df.drop(columns=[self.__anomaly_column])
        if self.__subspace_column is not None:
            self.__df_data = self.__df_data.drop(columns=[self.__subspace_column])

    def __set_subspaces(self):
        self.__subspaces = self.__df[self.__subspace_column]

    def __set_outlier_indices(self):
        self.__outlier_indices = np.array(self.__df.loc[self.__labels == 1, self.__anomaly_column].index).tolist()

    def __set_inlier_indices(self):
        self.__inlier_indices = np.array(self.__df.loc[self.__labels() == 0, self.__anomaly_column].index).tolist()

    def __set_labels(self):
        self.__labels = self.__df[self.__anomaly_column]

    def __set_scores(self):
        self.__scores = self.__df[self.__anomaly_column]

    # Getter Functions

    def get_data(self):
        return self.__df_data

    def get_subspaces(self):
        return self.__subspaces

    def get_outlier_indices(self):
        return self.__outlier_indices

    def get_inlier_indices(self):
        return self.__inlier_indices

    def get_df(self):
        return self.__df

    def get_labels(self):
        return self.__labels

    def get_scores(self):
        return self.__scores
