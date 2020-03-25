import os


class SettingsConfig:

    def __init__(self):
        self.__SETTINGS_KEY = 'settings'
        self.__DATASET_KEY = 'dataset'
        self.__OUTPUT_FOLDER_KEY = "output_folder"
        self.__FOLDS_KEY = "kfolds"
        self.__ANOMALY_COL_KEY = "is_anomaly_column"
        self.__SUBSPACE_COL_KEY = "subspace_column"
        self.__TOP_K_POINTS_TO_EXPLAIN = "top_k_points_to_explain"
        self.__default_output_folder = os.path.join("results")
        self.__settings_json_obj = None

    def setup(self, settings_json_obj):
        self.__settings_json_obj = settings_json_obj[self.__SETTINGS_KEY]

    def get_dataset_path(self):
        return self.__settings_json_obj[self.__DATASET_KEY]

    def get_output_folder(self):
        if self.__OUTPUT_FOLDER_KEY not in self.__settings_json_obj:
            return os.path.join(self.__default_output_folder, self.__DATASET_KEY)
        else:
            return self.__settings_json_obj[self.__OUTPUT_FOLDER_KEY]

    def
