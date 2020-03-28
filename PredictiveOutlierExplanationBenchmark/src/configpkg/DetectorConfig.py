class DetectorConfig:

    __DETECTOR_KEY = "detector"
    __ID_KEY = "id"
    __PARAMS_KEY = "params"
    __detector_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        DetectorConfig.__detector_json_obj = config_json_obj[DetectorConfig.__DETECTOR_KEY]

    @staticmethod
    def get_id():
        return DetectorConfig.__detector_json_obj[DetectorConfig.__ID_KEY]

    @staticmethod
    def get_params():
        return DetectorConfig.__detector_json_obj[DetectorConfig.__PARAMS_KEY]
