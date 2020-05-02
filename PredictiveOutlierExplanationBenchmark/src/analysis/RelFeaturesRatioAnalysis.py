from PredictiveOutlierExplanationBenchmark.src.utils import utils
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from PredictiveOutlierExplanationBenchmark.src.utils.pseudo_samples import PseudoSamplesMger
import json
import pandas as pd


class RelFeaturesRatio:

    def __init__(self, path_to_dir, metric_id):
        self.path_to_dir = path_to_dir
        self.metric_id = metric_id
        self.nav_files = None

    def analyze(self):
        self.nav_files = self.__read_nav_files()
        self.nav_files = self.__sort_files_by_dim()
        self.__analysis_per_nav_file()

    def __analysis_per_nav_file(self):
        for dim, nav_file in self.nav_files.items():
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=True)
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            optimal_features = utils.extract_optimal_features(original_dataset_path)
            metrics = self.__calculate_feature_metrics_per_k(ps_mger, optimal_features)
            print(dim, metrics)

    def __calculate_feature_metrics_per_k(self, ps_mger, optimal_features):
        feature_metrics_per_k = {}
        for k, best_model in ps_mger.best_model_per_k.items():
            selected_features = best_model['feature_selection']['features']
            features_prec = self.__features_precision(selected_features, optimal_features)
            features_recall = self.__features_recall(selected_features, optimal_features)
            feature_metrics_per_k[k] = {'features_precision': features_prec, 'features_recall': features_recall}
        return dict(sorted(feature_metrics_per_k.items()))

    def __features_precision(self, selected_features, optimal_features):
        return len(optimal_features.intersection(selected_features)) / len(selected_features)

    def __features_recall(self, selected_features, optimal_features):
        return len(optimal_features.intersection(selected_features)) / len(optimal_features)

    def __sort_files_by_dim(self):
        nav_files_sort_by_dim = {}
        for nfile in self.nav_files:
            data_dim = pd.read_csv(nfile[FileKeys.navigator_original_dataset_path]).shape[1]
            nav_files_sort_by_dim[data_dim] = nfile
        return dict(sorted(nav_files_sort_by_dim.items()))

    def __read_nav_files(self):
        nav_files = []
        nav_files_paths = utils.get_files_recursively(self.path_to_dir, FileNames.navigator_fname)
        for f in nav_files_paths:
            with open(f) as json_file:
                nav_files.append(json.load(json_file))
        return nav_files
