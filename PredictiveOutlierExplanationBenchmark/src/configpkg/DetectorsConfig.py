class DetectorConfig:

    __DETECTORS_KEY = "detectors"
    __ID_KEY = "id"
    __PARAMS_KEY = "params"
    __detectors_json_obj = None

    def __init__(self):
        pass

    @staticmethod
    def setup(config_json_obj):
        DetectorConfig.__detectors_json_obj = config_json_obj[DetectorConfig.__DETECTORS_KEY]

    @staticmethod
    def get_detectors_confs():
        return DetectorConfig.__detectors_json_obj

    @staticmethod
    def id_key():
        return DetectorConfig.__ID_KEY

    @staticmethod
    def detectors_key():
        return DetectorConfig.__DETECTORS_KEY

    @staticmethod
    def params_key():
        return DetectorConfig.__PARAMS_KEY
