import os


class SettingsConfig:

    __SETTINGS_KEY = 'settings'
    __TASK_KEY = 'task'
    __FOLDS_KEY = "kfolds"
    __TOP_K_POINTS_TO_EXPLAIN_KEY = "top_k_points_to_explain"
    __DETECTOR_KEY = 'detector'
    __PSEUDO_SAMPLES_KEY = 'pseudo_samples_per_outlier'
    __default_output_folder = os.path.join("old_results")
    __settings_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        SettingsConfig.__settings_json_obj = config_json_obj[SettingsConfig.__SETTINGS_KEY]

    @staticmethod
    def get_task():
        return 'classification'
        task = SettingsConfig.__settings_json_obj[SettingsConfig.__TASK_KEY]
        assert task == 'regression' or task == 'classification', "Task should be either 'regression' or 'classification'"
        return task

    @staticmethod
    def is_classification_task():
        return SettingsConfig.get_task() == 'classification'

    @staticmethod
    def get_kfolds():
        return SettingsConfig.__settings_json_obj[SettingsConfig.__FOLDS_KEY]

    @staticmethod
    def get_detector():
        if SettingsConfig.__DETECTOR_KEY not in SettingsConfig.__settings_json_obj:
            return None
        else:
            return SettingsConfig.__settings_json_obj[SettingsConfig.__DETECTOR_KEY]

    @staticmethod
    def get_top_k_points_to_explain():
        assert SettingsConfig.is_classification_task(), "Top-k points to explain applies only to classification task"
        return SettingsConfig.__settings_json_obj[SettingsConfig.__TOP_K_POINTS_TO_EXPLAIN_KEY]

    @staticmethod
    def get_pseudo_samples_array():
        if SettingsConfig.__PSEUDO_SAMPLES_KEY in SettingsConfig.__settings_json_obj:
            return SettingsConfig.__settings_json_obj[SettingsConfig.__PSEUDO_SAMPLES_KEY]
        else:
            return None

