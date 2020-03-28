class DatasetConfig:

    __DATASET_KEY = 'dataset'
    __DATASET_PATH_KEY = 'path'
    __ANOMALY_COL_KEY = 'is_anomaly_column'
    __SUBSPACE_COL_KEY = 'subspace_column'
    __dataset_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        DatasetConfig.__dataset_json_obj = config_json_obj[DatasetConfig.__DATASET_KEY]

    @staticmethod
    def get_dataset_path():
        return DatasetConfig.__dataset_json_obj[DatasetConfig.__DATASET_PATH_KEY]

    @staticmethod
    def get_anomaly_column_name():
        return DatasetConfig.__dataset_json_obj[DatasetConfig.__ANOMALY_COL_KEY]

    @staticmethod
    def get_subspace_column_name():
        if DatasetConfig.__SUBSPACE_COL_KEY not in DatasetConfig.__dataset_json_obj:
            return None
        else:
            return DatasetConfig.__dataset_json_obj[DatasetConfig.__SUBSPACE_COL_KEY]
