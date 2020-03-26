import os


class SettingsConfig:

    def __init__(self):
        self.__SETTINGS_KEY = 'settings'
        self.__TASK_KEY = 'task'
        self.__FOLDS_KEY = "kfolds"
        self.__TEST_SIZE_KEY = "test_size"
        self.__TOP_K_POINTS_TO_EXPLAIN_KEY = "top_k_points_to_explain"
        self.__PSEUDO_SAMPLES_KEY = 'pseudo_samples_per_outlier'
        self.__default_output_folder = os.path.join("results")
        self.__settings_json_obj = None

    def setup(self, settings_json_obj):
        self.__settings_json_obj = settings_json_obj[self.__SETTINGS_KEY]

    def get_task(self):
        return self.__settings_json_obj[self.__TASK_KEY]

    def task_is_classification(self):
        return self.get_task() == 'classification'

    def get_folds(self):
        return self.__settings_json_obj[self.__FOLDS_KEY]

    def get_test_size(self):
        return self.__settings_json_obj[self.__TEST_SIZE_KEY]

    def get_top_k_points_to_explain(self):
        assert self.__TASK_KEY == 'regression' or self.__TASK_KEY == 'classification'
        if self.__TASK_KEY == 'regression':
            return None
        else:
            return self.__settings_json_obj[self.__TOP_K_POINTS_TO_EXPLAIN_KEY]

    def get_pseudo_samples_num(self):
        return self.__settings_json_obj[self.__PSEUDO_SAMPLES_KEY]
