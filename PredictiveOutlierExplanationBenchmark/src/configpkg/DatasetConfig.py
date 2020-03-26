class DatasetConfig:

    def __init__(self):
        self.__DATASET_KEY = 'dataset'
        self.__DATASET_PATH_KEY = 'path'
        self.__ANOMALY_COL_KEY = 'is_anomaly_column'
        self.__SUBSPACE_COL_KEY = 'subspace_column'
        self.__dataset_json_obj = None

    def setup(self, dataset_json_obj):
        self.__dataset_json_obj = dataset_json_obj[self.__DATASET_KEY]

    def get_dataset_path(self):
        return self.__dataset_json_obj[self.__DATASET_PATH_KEY]

    def get_anomaly_column(self):
        return self.__dataset_json_obj[self.__ANOMALY_COL_KEY]

    def get_subspace_column(self):
        if self.__SUBSPACE_COL_KEY not in self.__dataset_json_obj:
            return None
        else:
            return self.__dataset_json_obj[self.__SUBSPACE_COL_KEY]
