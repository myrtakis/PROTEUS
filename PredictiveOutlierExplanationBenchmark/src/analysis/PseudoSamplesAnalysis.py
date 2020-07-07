import warnings

from matplotlib.font_manager import FontProperties

warnings.filterwarnings("ignore")
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)

from utils.helper_functions import read_nav_files, sort_files_by_dim
from utils.shared_names import FileKeys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


class PseudoSamplesAnalysis:

    def __init__(self, path_to_dir, real=True):
        self.path_to_dir = path_to_dir
        self.nav_files = None
        self.real = real

    def analyze(self):
        print('Analyze results in path', self.path_to_dir)
        self.nav_files = read_nav_files(self.path_to_dir)
        self.__pseudo_samples_analysis()

    def __pseudo_samples_analysis(self):
        self.nav_files = sort_files_by_dim(self.nav_files)
        ps_samples_outlier_ratio_per_k = {}
        best_dets = {}
        for dim, nav_file in self.nav_files.items():
            last_original_point_ind = self.__get_last_orginal_point_index(nav_file)
            best_dets[dim] = self.__get_best_detector(nav_file)
            assert last_original_point_ind is not None
            pseudo_samples_data = self.__datasets_with_pseudo_samples(nav_file[FileKeys.navigator_pseudo_samples_key])
            ps_samples_outlier_ratio_per_k = self.__update_pseudo_samples_outlier_ratio(ps_samples_outlier_ratio_per_k,
                                                                                        pseudo_samples_data,
                                                                                        last_original_point_ind)
        dims_dets = []
        for k in self.nav_files:
            if self.real:
                dims_dets.append(str(k-1) + '-d') # (' + best_dets[k] + ')')
            else:
                dims_dets.append(str(k - 2) + '-d')

        self.__plot_info_as_table(dims_dets, ps_samples_outlier_ratio_per_k)

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

    def __get_last_orginal_point_index(self, nav_file):
        for id, data in nav_file[FileKeys.navigator_pseudo_samples_key].items():
            if data[FileKeys.navigator_pseudo_samples_num_key] == 0:
                return pd.read_csv(data[FileKeys.navigator_pseudo_samples_data_path]).shape[0]

    def __get_best_detector(self, nav_file):
        det_info_file = nav_file[FileKeys.navigator_detector_info_path]
        with open(det_info_file) as json_file:
            dets = json.load(json_file)
            best_det = None
            best_det_effect = None
            metric = 'roc_auc'
            for d_id, data in dets.items():
                effect = data['effectiveness'][metric]
                if best_det_effect is None or best_det_effect < effect:
                    best_det_effect = effect
                    best_det = data['id']
            return best_det

    def __plot_info_as_table(self, cols, data):

        indexes = ['K = ' + str(k) for k in data.keys()]
        ratios = list(data.values())
        df = pd.DataFrame(ratios, columns=cols, index=indexes)

        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                         loc='center', rowLabels=df.index,
                         cellLoc='center')

        table.scale(0.8, 2.5)

        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.show()

if __name__ == '__main__':
    # path = '../results_normal/random_oversampling/iforest/classification/datasets/synthetic/hics/group_g1'
    # path = '../results_predictive/random_oversampling/iforest/classification/datasets/synthetic/hics/group_g1'
    # path = '../results_normal/random_oversampling/lof/classification/datasets/real'
    # path = '../results_predictive/random_oversampling/lof/classification/datasets/real'

    path, real = '../results_predictive/lof/protean/random_oversampling/classification/datasets/synthetic', False

    PseudoSamplesAnalysis(path, real=real).analyze()
