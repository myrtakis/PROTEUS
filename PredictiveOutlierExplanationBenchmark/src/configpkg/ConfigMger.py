import json
from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.DetectorConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.ClassifiersConfig import *
from PredictiveOutlierExplanationBenchmark.src.configpkg.FeatureSelectionConfig import *


class ConfigMger:

    def __init__(self):
        self.__SettingsConf = SettingsConfig()
        self.__DetectorConf = DetectorConfig()
        self.__ClassifiersConf = ClassifiersConfig()
        self.__FeatureSelectionConf = FeatureSelectionConfig()
        self.__config_path = None
        self.__config_data = None

    # Base Functions

    def parse_json_config_file(self, path):
        self.__config_path = path
        with open(path) as json_file:
            self.__config_data = json.load(json_file)

    def setup_configs(self):
        self.__SettingsConf.setup(self.__config_data[self.__SETTINGS_KEY])
        self.__DetectorConf.setup(self.__config_data[self.__DETECTOR_KEY])
        self.__ClassifiersConf.setup(self.__config_data[self.__CLASSIFIERS_KEY])
        self.__FeatureSelectionConf.setup(self.__config_data[self.__FEATURE_SELECTION_KEY])

    # Getter Functions

    def get_config_path(self):
        return self.__config_path

    # Util Functions

    def model_configurations(self):
        pass
