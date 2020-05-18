from pathlib import Path

from PredictiveOutlierExplanationBenchmark.src.utils import helper_functions
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from PredictiveOutlierExplanationBenchmark.src.utils.pseudo_samples import PseudoSamplesMger
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PredictiveOutlierExplanationBenchmark.src.utils.helper_functions import read_nav_files, sort_files_by_dim
import os

class PerfAnalysis:

    __markers = ['s', 'o', 'd', 'v', '^', '<', '>', 'x', '+']

    def __init__(self, path_to_dir, metric_id, hold_out_effectiveness=False):
        self.path_to_dir = path_to_dir
        self.metric_id = metric_id
        if hold_out_effectiveness:
            self.effectiveness_key = 'hold_out_effectiveness'
        else:
            self.effectiveness_key = 'effectiveness'
        self.nav_files = None

    def analyze(self, original_data_analysis=False, real_data=False):
        self.nav_files = read_nav_files(self.path_to_dir)
        if real_data:
            self.__analyze_real_datasets(original_data_analysis)
        else:
            self.__analyze_synthetic_datasets(original_data_analysis)

    def __analyze_synthetic_datasets(self, original_data_analysis=False):
        self.nav_files = sort_files_by_dim(self.nav_files)
        if original_data_analysis:
            print('performance on original data without feature selection')
            self.__analysis_of_original_synthetic_data(fs=False)
            print('performance on original data with feature selection')
            self.__analysis_of_original_synthetic_data(fs=True)
        print('Learning the boundary')
        self.__analysis_per_nav_file_synthetic_data(fs=False)
        print('Explaining the boundary')
        self.__analysis_per_nav_file_synthetic_data(fs=True)

    def __analyze_real_datasets(self, original_data_analysis=False):
        self.nav_files = sort_files_by_dim(self.nav_files)
        if original_data_analysis:
            print('performance on original data without feature selection')
            self.__analysis_of_original_real_data(fs=False)
            print('performance on original data with feature selection')
            self.__analysis_of_original_real_data(fs=True)
        print('Learning the boundary')
        self.__analysis_per_nav_file_real_data(fs=False)
        print('Explaining the boundary')
        self.__analysis_per_nav_file_real_data(fs=True)

    def __analysis_of_original_synthetic_data(self, fs):
        rel_fratio_perfs_by_orig = {}
        for dim, nav_file in self.nav_files.items():
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            rel_fratio = self.__compute_rel_fratio(original_dataset_path, dim - 2)
            original_data = nav_file[FileKeys.navigator_original_data]
            rel_fratio_perfs_by_orig[rel_fratio] = helper_functions.get_best_model_perf_original_data(original_data, self.metric_id, fs)
        print(rel_fratio_perfs_by_orig)

    def __analysis_of_original_real_data(self, fs):
        for nav_file in self.nav_files:
            original_data = nav_file[FileKeys.navigator_original_data]
            best_model_perf, best_conf = helper_functions.get_best_model_perf_original_data(original_data, self.metric_id, fs)
            print(best_model_perf, best_conf)

    def __analysis_per_nav_file_synthetic_data(self, fs):
        rel_fratio_perfs_by_k = {}
        rel_fratio_dims = {}
        for dim, nav_file in self.nav_files.items():
            original_dataset_path = nav_file[FileKeys.navigator_original_dataset_path]
            rel_fratio = self.__compute_rel_fratio(original_dataset_path, dim-2)
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=fs)
            perf_per_k, best_model_ids = self.__calculate_performance_per_k(ps_mger)
            rel_fratio_perfs_by_k[rel_fratio] = perf_per_k
            rel_fratio_dims[rel_fratio] = dim - 2
        self.__plot_relf_ratio_to_perf(rel_fratio_perfs_by_k, rel_fratio_dims, fs)

    def __analysis_per_nav_file_real_data(self, fs):
        perf_by_dim = {}
        names_by_dim = {}
        for dim, nav_file in self.nav_files.items():
            original_file_path = nav_file[FileKeys.navigator_original_dataset_path]
            dataset_name = os.path.splitext(os.path.basename(original_file_path))[0]
            names_by_dim[dim] = dataset_name
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=fs)
            perf_per_k, best_model_ids = self.__calculate_performance_per_k(ps_mger)
            for k, perf in perf_per_k.items():
                perf_by_dim.setdefault(k, {})
                perf_by_dim[k][dim-1] = perf
        for k, data in perf_by_dim.items():
            perf_by_dim[k] = dict(sorted(data.items()))
        self.__plot_real_data_by_dim(perf_by_dim, names_by_dim, fs)

    def __calculate_performance_per_k(self, ps_mger):
        perf_per_k = {}
        best_model_ids = set()
        for k, best_model in ps_mger.best_model_per_k.items():
            perf_per_k[k] = best_model[self.effectiveness_key]
            best_model_ids.add(best_model['feature_selection']['id'] + '_' + best_model['classifiers']['id'])
        return perf_per_k, best_model_ids

    def __compute_rel_fratio(self, original_dataset_path, dim):
        optimal_features = helper_functions.extract_optimal_features(original_dataset_path)
        return round((len(optimal_features) / dim) * 100)

    def __plot_relf_ratio_to_perf_orig(self, rel_fratio_perfs_orig, fs):
        x_labels = [str(i) + '%' for i in rel_fratio_perfs_orig.keys()]
        plt.plot(list(rel_fratio_perfs_orig.keys()), list(rel_fratio_perfs_orig.vales()))
        plt.xticks(range(len(x_labels)), x_labels)
        plt.legend()
        plt.xlabel('Relevant Feature Ratio')
        plt.ylabel(self.metric_id)
        title = 'Explanation of Decision Boundary' if fs is True else 'Learning of Decision Boundary'
        plt.title(title)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        plt.clf()

    def __plot_real_data_by_dim(self, perf_by_dim, names_by_dim, fs):
        x_labels = [n + ' (' + str(d) + '-d)' for d, n in names_by_dim.items()]
        i = 0
        for k, perfs in perf_by_dim.items():
            plt.plot(range(len(perfs.keys())), list(perfs.values()), label='K = ' + str(k), marker=PerfAnalysis.__markers[i])
            i += 1
        plt.xticks(range(len(x_labels)), x_labels, rotation=45)
        plt.legend()
        plt.xlabel('Datasets (dimensions)')
        plt.ylabel(self.metric_id)
        title = 'Explanation of Decision Boundary' if fs is True else 'Learning of Decision Boundary'
        plt.title(title)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        plt.clf()

    def __plot_relf_ratio_to_perf(self, rel_fratio_perfs_by_k,  rel_fratio_dims, fs):
        k_rel_fratio = {}
        for rel_fratio, data in rel_fratio_perfs_by_k.items():
            for k, perf in data.items():
                k_rel_fratio.setdefault(k, {})
                k_rel_fratio[k][rel_fratio] = perf
        x_labels = []
        for rel_fratio in rel_fratio_perfs_by_k.keys():
            # x_labels.append(str(rel_fratio_dims[rel_fratio]) + '-d (' + str(rel_fratio) + '%)')
            x_labels.append(str(rel_fratio_dims[rel_fratio]) + '-d')
        i = 0
        for k, data in k_rel_fratio.items():
            plt.plot(range(len(data.keys())), list(data.values()), label='K = ' + str(k), marker=PerfAnalysis.__markers[i])
            i += 1
        plt.xticks(range(len(x_labels)), x_labels)
        plt.legend()
       # plt.xlabel('Dataset Dimensionality (Relevant Feature Ratio %)')
        plt.xlabel('Dataset Dimensionality')
        plt.ylabel(self.metric_id)
        title = 'Explanation of Decision Boundary' if fs is True else 'Learning of Decision Boundary'
        plt.title(title)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        plt.clf()
