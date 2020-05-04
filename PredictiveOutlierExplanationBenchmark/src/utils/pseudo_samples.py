from pathlib import Path
import json
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from PredictiveOutlierExplanationBenchmark.src.utils.utils import fs_key


class PseudoSamplesMger:

    def __init__(self, pseudo_samples_raw, metric_id, fs):
        self.pseudo_samples_raw = pseudo_samples_raw
        self.metric_id = metric_id
        self.fs = fs
        self.best_model_per_k = {}
        self.best_k = None
        self.best_k_id = None
        self.__init_best_samples_best_models()

    def __init_best_samples_best_models(self):
        for ps_samples, ps_data in self.pseudo_samples_raw.items():
            k = ps_data[FileKeys.navigator_pseudo_samples_num_key]
            best_model = self.__get_best_model_from_dir(ps_data[FileKeys.navigator_pseudo_sample_dir_key])
            self.best_model_per_k[k] = best_model

    def __get_best_model_from_dir(self, ps_dir):
        best_model_file = Path(ps_dir, FileNames.best_model_fname)
        with open(best_model_file) as json_file:
            return json.load(json_file)[fs_key(self.fs)][self.metric_id]

    def get_best_model(self, stat_significant=False):
        best_model = None
        best_k = None
        for k, kmodel in self.best_model_per_k.items():
            if best_model is None or self.get_effectiveness_of_best_model(best_model) < self.get_effectiveness_of_best_model(kmodel):
                best_model = kmodel
                best_k = k
        return best_model, best_k

    def optimal_pseudo_samples(self):
        self.sort_ps_samples_by_perf()
        # return something
        pass

    def sort_ps_samples_by_perf(self):
        # return something
        pass

    def get_effectiveness_of_best_model(self, best_model):
        return best_model[fs_key(self.fs)][self.metric_id]['effectiveness']
