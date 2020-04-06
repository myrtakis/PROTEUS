class ModelConf:

    def __init__(self, fsel, clf, conf_id):
        assert fsel is not None
        assert clf is not None
        assert conf_id is not None
        self.__conf_id = conf_id
        self.__fsel = fsel
        self.__clf = clf
        self.__metric_id = None
        self.__effectiveness = None

    def set_effectiveness(self, value, metric_id, conf_id):
        assert value is not None
        assert metric_id is not None and metric_id != '', metric_id
        assert self.__metric_id is None or self.__metric_id == metric_id, self.__metric_id + ' ' + metric_id
        assert self.__conf_id == conf_id, str(self.__conf_id) + ' ' + str(conf_id)
        if isinstance(value, dict):
            value = float(list(value.values())[0])
        self.__metric_id = metric_id
        self.__effectiveness = value

    def get_fsel(self):
        assert self.__fsel is not None
        return self.__fsel

    def get_clf(self):
        assert self.__clf is not None
        return self.__clf

    def get_metric_id(self):
        assert self.__metric_id is not None
        return self.__metric_id

    def get_conf_id(self):
        assert self.__conf_id is not None
        return self.__conf_id

    def get_effectiveness(self):
        assert self.__effectiveness is not None
        return self.__effectiveness
