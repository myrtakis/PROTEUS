from PredictiveOutlierExplanationBenchmark.src.utils import utils
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DetectorAnalysis:

    __markers = ['s', 'o', 'd', 'v', '^', '<', '>', 'x', '+']

    def __init__(self, path_to_dir, metric_id):
        self.path_to_dir = path_to_dir
        self.metric_id = metric_id
        self.nav_files = None

    def analyze(self, real_data=False):
        self.nav_files = self.__read_nav_files()
        if real_data:
            self.__analyze_real_data()
        else:
            self.__analyze_synthetic_data()

    def __analyze_real_data(self):
        det_perfs = self.__detectors_perfs_real_data()
        print(det_perfs)

    def __analyze_synthetic_data(self):
        self.nav_files = self.__sort_files_by_dim()
        det_perfs = self.__detectors_perfs_synthetic_data()
        self.__plot_detectors_perfs(det_perfs)

    def __plot_detectors_perfs(self, detectors_perfs):
        rel_fratio = []
        i = 0
        for det, data in detectors_perfs.items():
            rel_fratio = list(data.keys())
            plt.plot(range(len(rel_fratio)), list(data.values()), marker=DetectorAnalysis.__markers[i], label=det)
            i += 1
        x_labels = [str(i) + '%' for i in rel_fratio]
        plt.xticks(range(len(x_labels)), x_labels)
        plt.legend()
        plt.xlabel('Relevant Feature Ratio')
        plt.ylabel(self.metric_id)
        title = 'Outlier Detectors'
        plt.title(title)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        plt.clf()

    def __detectors_perfs_synthetic_data(self):
        det_perfs = {}
        for dim, nav_file in self.nav_files.items():
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            rel_fratio = self.__compute_rel_fratio(original_dataset_path, dim - 2)
            detectors_info_file = nav_file[FileKeys.navigator_detector_info_path]
            with open(detectors_info_file) as json_file:
                for det, data in json.load(json_file).items():
                    det_perfs.setdefault(det, {})
                    det_perfs[det][rel_fratio] = data['effectiveness'][self.metric_id]
        return det_perfs

    def __detectors_perfs_real_data(self):
        det_perfs = {}
        for nav_file in self.nav_files:
            detectors_info_file = nav_file[FileKeys.navigator_detector_info_path]
            with open(detectors_info_file) as json_file:
                for det, data in json.load(json_file).items():
                    det_perfs[det] = data['effectiveness'][self.metric_id]
        return det_perfs

    def __compute_rel_fratio(self, original_dataset_path, dim):
        optimal_features = utils.extract_optimal_features(original_dataset_path)
        return round((len(optimal_features) / dim) * 100)

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
