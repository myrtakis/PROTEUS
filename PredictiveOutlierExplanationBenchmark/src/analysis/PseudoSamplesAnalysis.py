from PredictiveOutlierExplanationBenchmark.src.utils.helper_functions import read_nav_files, sort_files_by_dim
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileKeys
import pandas as pd
import numpy as np


class PseudoSamplesAnalysis:

    def __init__(self, path_to_dir):
        self.path_to_dir = path_to_dir
        self.nav_files = None

    def analyze(self):
        print('Analyze results in path', self.path_to_dir)
        self.nav_files = read_nav_files(self.path_to_dir)
        self.__pseudo_samples_analysis()

    def __pseudo_samples_analysis(self):
        self.nav_files = sort_files_by_dim(self.nav_files)
        ps_samples_outlier_ratio_per_k = {}
        print(self.nav_files.keys())
        for dim, nav_file in self.nav_files.items():
            last_original_point_ind = self.__get_last_orginal_point_index(nav_file[FileKeys.navigator_pseudo_samples_key])
            pseudo_samples_data = self.__datasets_with_pseudo_samples(nav_file[FileKeys.navigator_pseudo_samples_key])
            ps_samples_outlier_ratio_per_k = self.__update_pseudo_samples_outlier_ratio(ps_samples_outlier_ratio_per_k,
                                                                                        pseudo_samples_data,
                                                                                        last_original_point_ind)
        print(ps_samples_outlier_ratio_per_k)

    def __datasets_with_pseudo_samples(self, pseudo_samples_data):
        ps_samples_data_dict = {}
        for ps_samples, ps_data in pseudo_samples_data.items():
            num_ps_samples = ps_data[FileKeys.navigator_pseudo_samples_num_key]
            if num_ps_samples == 0:
                continue
            df = pd.read_csv(ps_data[FileKeys.navigator_pseudo_samples_data_path])
            ps_samples_data_dict[num_ps_samples] = df
        return ps_samples_data_dict

    def __update_pseudo_samples_outlier_ratio(self, pseudo_samples_outlier_ratio_per_k, pseudo_samples_data, last_original_point_ind):
        for ps_num, ps_df in pseudo_samples_data.items():
            pseudo_samples_outlier_ratio_per_k.setdefault(ps_num, [])
            ps_df_cropped = ps_df.iloc[last_original_point_ind:, :]
            outliers = np.where(ps_df_cropped['is_anomaly'] == 1)[0]
            ps_outlier_ratio = round(float(len(outliers)) / float(ps_df_cropped.shape[0]), 2)
            pseudo_samples_outlier_ratio_per_k[ps_num].append(ps_outlier_ratio)
        return pseudo_samples_outlier_ratio_per_k

    def __get_last_orginal_point_index(self, pseudo_samples_data):
        for ps_samples, ps_data in pseudo_samples_data.items():
            num_ps_samples = ps_data[FileKeys.navigator_pseudo_samples_num_key]
            if num_ps_samples == 0:
                return pd.read_csv(ps_data[FileKeys.navigator_pseudo_samples_data_path]).shape[0]


if __name__ == '__main__':
    path_synth = '../results/classification/datasets/synthetic/hics/group_g1'
    path = '../results/classification/datasets/real/arrhythmia_015'
    path_synth_pred = '../results_predictive/classification/datasets/synthetic/hics/group_g1'
    path = '../results_predictive/classification/datasets/real/arrhythmia_015'

    PseudoSamplesAnalysis(path_synth).analyze()

    PseudoSamplesAnalysis(path_synth_pred).analyze()
