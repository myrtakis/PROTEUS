class DetectorConfig:

    def __init__(self):
        self.__DETECTOR_KEY = "detector"
        self.__ID_KEY = "id"
        self.__PARAMS_KEY = "params"
        self.__detector_json_obj = None

    def setup(self, detector_json_obj):
        self.__detector_json_obj = detector_json_obj[self.__DETECTOR_KEY]