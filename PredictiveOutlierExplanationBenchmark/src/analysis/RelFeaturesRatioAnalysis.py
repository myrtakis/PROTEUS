from PredictiveOutlierExplanationBenchmark.src.utils import helper_functions
from PredictiveOutlierExplanationBenchmark.src.utils.helper_functions import sort_files_by_dim, read_nav_files
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from PredictiveOutlierExplanationBenchmark.src.utils.pseudo_samples import PseudoSamplesMger


class RelFeaturesRatio:

    def __init__(self, path_to_dir, metric_id):
        self.path_to_dir = path_to_dir
        self.metric_id = metric_id
        self.nav_files = None

    def analyze(self):
        self.nav_files = read_nav_files(self.path_to_dir)
        self.nav_files = sort_files_by_dim(self.nav_files)
        self.__analysis_per_nav_file()
        self.__analysis_of_original_data()

    def __analysis_of_original_data(self):
        print('Feature analysis of original data')
        for dim, nav_file in self.nav_files.items():
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            optimal_features = helper_functions.extract_optimal_features(original_dataset_path)
            original_data = nav_file[FileKeys.navigator_original_data]
            selected_features, conf_id = helper_functions.get_best_model_features_original_data(original_data, self.metric_id)
            features_prec = self.__features_precision(selected_features, optimal_features)
            features_recall = self.__features_recall(selected_features, optimal_features)
            print(dim, 'Pred:', features_prec, 'Recall:', features_recall, 'conf', conf_id, '#Feautures', len(selected_features))

    def __analysis_per_nav_file(self):
        #todo BUG: In feature precision and recall you should consider only the relevant features of true detected
        # outliers and not for all of them
        print('Feature analysis per navigator file')
        for dim, nav_file in self.nav_files.items():
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=True)
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            optimal_features = helper_functions.extract_optimal_features(original_dataset_path)
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

