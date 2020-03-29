import json
from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import SettingsConfig
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorConfig import DetectorConfig
from PredictiveOutlierExplanationBenchmark.src.configpkg.DatasetConfig import DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import ClassifiersConfig
from PredictiveOutlierExplanationBenchmark.src.configpkg.FeatureSelectionConfig import FeatureSelectionConfig


class ConfigMger:

    __config_path = None
    __config_data = None

    def __init__(self):
        pass

    # Base Functions
    @staticmethod
    def setup_configs(path):
        with open(path) as json_file:
            ConfigMger.__config_data = json.load(json_file)
        SettingsConfig.setup(ConfigMger.__config_data)
        DetectorConfig.setup(ConfigMger.__config_data)
        ClassifiersConfig.setup(ConfigMger.__config_data)
        FeatureSelectionConfig.setup(ConfigMger.__config_data)
        DatasetConfig.setup(ConfigMger.__config_data)

    # Getter Functions

    @staticmethod
    def get_config_path():
        return ConfigMger.__config_path

    @staticmethod
    def get_config_data():
        return ConfigMger.__config_data
