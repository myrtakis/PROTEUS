from utils import helper_functions
from utils.shared_names import *
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.helper_functions import get_best_detector_from_info_file, \
    read_nav_files, sort_files_by_dim


class DetectorAnalysis:

    __markers = ['s', 'o', 'd', 'v', '^', '<', '>', 'x', '+']

    def __init__(self, path_to_dir, metric_id):
        self.path_to_dir = path_to_dir
        self.metric_id = metric_id
        self.nav_files = None

    def analyze(self, real_data=False):
        self.nav_files = read_nav_files(self.path_to_dir)
        if real_data:
            self.__analyze_real_data()
        else:
            self.__analyze_synthetic_data()

    def analyze_hold_out_effectiveness(self):
        print('Hold out effectiveness')
        self.nav_files = read_nav_files(self.path_to_dir)
        for nav_file in self.nav_files:
            detectors_info_file = nav_file[FileKeys.navigator_detector_info_path]
            original_dataset = nav_file[FileKeys.navigator_original_dataset_path]
            with open(detectors_info_file) as json_file:
                det_file = json.load(json_file)
                best_det, train_perf = get_best_detector_from_info_file(detectors_info_file, self.metric_id)
                hold_out_perf = det_file[best_det]['hold_out_effectiveness'][self.metric_id]
            print(original_dataset, '\t', best_det, round(train_perf, 2), round(hold_out_perf, 2))

    def __analyze_real_data(self):
        det_perfs = self.__detectors_perfs_real_data()
        print(det_perfs)

    def __analyze_synthetic_data(self):
        self.nav_files = sort_files_by_dim(self.nav_files)
        det_perfs, rel_fratio_dims = self.__detectors_perfs_synthetic_data()
        self.__plot_detectors_perfs(det_perfs, rel_fratio_dims)

    def __plot_detectors_perfs(self, detectors_perfs, rel_fratio_dims):
        rel_fratio = []
        i = 0
        for det, data in detectors_perfs.items():
            rel_fratio = list(data.keys())
            plt.plot(range(len(rel_fratio)), list(data.values()), marker=DetectorAnalysis.__markers[i], label=det)
            i += 1
        x_labels = []
        for k in rel_fratio:
            x_labels.append(str(rel_fratio_dims[k]) + '-d (' + str(k) + '%)')
        plt.xticks(range(len(x_labels)), x_labels)
        plt.legend()
        plt.xlabel('Dataset Dimensionality (Relevant Feature Ratio %)')
        plt.ylabel(self.metric_id)
        title = 'Outlier Detectors'
        plt.title(title)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        plt.clf()

    def __detectors_perfs_synthetic_data(self):
        det_perfs = {}
        rel_fratio_dims = {}
        for dim, nav_file in self.nav_files.items():
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            rel_fratio = self.__compute_rel_fratio(original_dataset_path, dim - 2)
            rel_fratio_dims[rel_fratio] = dim - 2
            detectors_info_file = nav_file[FileKeys.navigator_detector_info_path]
            with open(detectors_info_file) as json_file:
                for det, data in json.load(json_file).items():
                    det_perfs.setdefault(det, {})
                    det_perfs[det][rel_fratio] = data['effectiveness'][self.metric_id]
        return det_perfs, rel_fratio_dims

    def __detectors_perfs_real_data(self):
        det_perfs = {}
        for nav_file in self.nav_files:
            detectors_info_file = nav_file[FileKeys.navigator_detector_info_path]
            with open(detectors_info_file) as json_file:
                for det, data in json.load(json_file).items():
                    det_perfs[det] = data['effectiveness'][self.metric_id]
        return det_perfs

    def __compute_rel_fratio(self, original_dataset_path, dim):
        optimal_features = helper_functions.extract_optimal_features(original_dataset_path)
        return round((len(optimal_features) / dim) * 100)

