import json
from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.FeatureSelectionConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.DatasetConfig import *


class ConfigMger:

    def __init__(self, path):
        self.__SettingsConf = SettingsConfig()
        self.__DetectorConf = DetectorConfig()
        self.__ClassifiersConf = ClassifiersConfig()
        self.__FeatureSelectionConf = FeatureSelectionConfig()
        self.__DatasetConf = DatasetConfig()
        self.__config_path = path
        with open(path) as json_file:
            self.__config_data = json.load(json_file)

    # Base Functions

    def setup_configs(self):
        self.__SettingsConf.setup(self.__config_data)
        self.__DetectorConf.setup(self.__config_data)
        self.__ClassifiersConf.setup(self.__config_data)
        self.__FeatureSelectionConf.setup(self.__config_data)
        self.__DatasetConf.setup(self.__config_data)

    # Getter Functions

    def get_config_path(self):
        return self.__config_path

    def get_config_data(self):
        return self.__config_data

    def get_SettingsConf(self):
        return self.__SettingsConf

    def get_DetectorConf(self):
        return self.__DetectorConf

    def get_ClassifiersConf(self):
        return self.__ClassifiersConf

    def get_FeatureSelectionConf(self):
        return self.__FeatureSelectionConf

    def get_DatasetConf(self):
        return self.__DatasetConf

    # Util Functions

    def model_configurations(self):
        pass
