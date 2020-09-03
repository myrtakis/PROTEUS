from pathlib import Path
import json
from utils.shared_names import *
from utils.helper_functions import fs_key


class PseudoSamplesMger:

    def __init__(self, pseudo_samples_raw, metric_id, fs):
        self.pseudo_samples_raw = pseudo_samples_raw
        self.metric_id = metric_id
        self.fs = fs
        self.best_model_per_k = {}
        self.k_configurations = []
        self.dir_per_k = {}
        self.best_k = None
        self.best_k_id = None
        self.__init_best_samples_best_models()

    def __init_best_samples_best_models(self):
        for ps_samples, ps_data in self.pseudo_samples_raw.items():
            k = ps_data[FileKeys.navigator_pseudo_samples_num_key]
            self.k_configurations.append(k)
            best_model = self.__get_best_model_from_dir(ps_data[FileKeys.navigator_pseudo_sample_dir_key])
            self.best_model_per_k[k] = best_model
            self.dir_per_k[k] = ps_data[FileKeys.navigator_pseudo_sample_dir_key]
        assert len(self.k_configurations) == len(self.pseudo_samples_raw)

    def __get_best_model_from_dir(self, ps_dir):
        best_model_file = Path(ps_dir, FileNames.best_model_fname)
        with open(best_model_file, 'r') as json_file:
            return json.load(json_file)[fs_key(self.fs)][self.metric_id]

    def get_best_model(self, stat_significant=False):
        best_model = None
        best_k = None
        for k, kmodel in self.best_model_per_k.items():
            if best_model is None or best_model['effectiveness'] < kmodel['effectiveness']:
                best_model = kmodel
                best_k = k
        return best_model, best_k

    def get_best_model_per_k(self):
        return self.best_model_per_k

    def get_path_of_best_model(self):
        _, best_k = self.get_best_model()
        return self.dir_per_k[best_k]

    def optimal_pseudo_samples(self):
        self.sort_ps_samples_by_perf()
        # return something
        pass

    def sort_ps_samples_by_perf(self):
        # return something
        pass

    def list_k_confs(self):
        return self.k_configurations

    def get_info_field_of_k(self, k, data_path_key):
        for key, data in self.pseudo_samples_raw.items():
            if data[FileKeys.navigator_pseudo_samples_num_key] == k:
                return data[data_path_key]

