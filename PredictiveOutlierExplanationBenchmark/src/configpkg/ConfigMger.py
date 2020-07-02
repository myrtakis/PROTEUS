import json
from configpkg.SettingsConfig import SettingsConfig
from configpkg.DetectorsConfig import DetectorConfig
from configpkg.DatasetConfig import DatasetConfig
from configpkg.ClassifiersConfig import ClassifiersConfig
from configpkg.FeatureSelectionConfig import FeatureSelectionConfig


class ConfigMger:

    __config_path = None
    __config_data = None

    def __init__(self):
        pass

    # Base Functions
    @staticmethod
    def setup_configs(path):
        ConfigMger.__config_path = path
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
