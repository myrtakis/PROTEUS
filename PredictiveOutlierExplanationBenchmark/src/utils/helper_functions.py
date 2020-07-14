import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from pipeline.DatasetTransformer import Transformer
from utils.shared_names import *
import pandas as pd
import os
from holders.Dataset import Dataset


def get_files_recursively(path_to_dir, contain_filter):
    if not os.path.isdir(path_to_dir):
        return [path_to_dir]
    paths = []
    for r, d, f in os.walk(path_to_dir):
        for file in f:
            if contain_filter in file:
                paths.append(os.path.join(r, file))
    return paths


def extract_optimal_features(dataset_path):
    df = pd.read_csv(dataset_path)
    subspaces_as_str = set(df.loc[df['subspaces'] != '-', 'subspaces'].values)
    optimal_features = set()
    for s in subspaces_as_str:
        optimal_features = optimal_features.union([int(f) for f in s[s.index('[') + 1: s.index(']')].split()])
    return optimal_features


def get_best_model_original_data(original_data_results_path, metric_id, fs):
    best_model_orig = Path(original_data_results_path, FileNames.best_model_fname)
    with open(best_model_orig) as json_file:
        best_model = json.load(json_file)[fs_key(fs)][metric_id]
        return best_model


def get_best_model_perf_original_data(original_data_results_path, metric_id, fs):
    best_model_orig = Path(original_data_results_path, FileNames.best_model_fname)
    with open(best_model_orig) as json_file:
        best_model = json.load(json_file)[fs_key(fs)][metric_id]
        fsel_id = best_model['feature_selection']['id']
        clf_id = best_model['classifiers']['id']
        if fs:
            best_conf = fsel_id + '_' + clf_id
        else:
            best_conf = clf_id
        return best_model['effectiveness'], best_conf


def get_best_detector_from_info_file(detectors_info_file_path, metric_id):
    with open(detectors_info_file_path) as json_file:
        max_perf = None
        best_det = None
        for det, data in json.load(json_file).items():
            if max_perf is None or max_perf < data['effectiveness'][metric_id]:
                max_perf = data['effectiveness'][metric_id]
                best_det = det
        return best_det, max_perf


def get_best_model_features_original_data(original_data_results_path, metric_id):
    best_model_orig = Path(original_data_results_path, FileNames.best_model_fname)
    with open(best_model_orig) as json_file:
        best_model = json.load(json_file)[fs_key(True)][metric_id]
        fsel_id = best_model['feature_selection']['id']
        features = best_model['feature_selection']['features']
        clf_id = best_model['classifiers']['id']
        best_conf = fsel_id + '_' + clf_id
        return features, best_conf


def add_noise_to_data(dataset, out_dims=None):
    np.random.seed(0)
    if out_dims is not None:
        out_dims_final = np.abs(out_dims - dataset.get_X().shape[1])
        out_shape = (dataset.get_X().shape[0], out_dims_final)
    else:
        out_shape = dataset.get_X().shape
    noise_data = np.random.normal(0, 1, out_shape)
    dataset_noise = pd.concat([dataset.get_df(), pd.DataFrame(noise_data)], axis=1)
    return Dataset(dataset_noise, dataset.get_anomaly_column_name(), dataset.get_subspace_column_name())


def add_datasets_oversampling(oversampling_method, dataset_detected_outliers, detector, threshold,
                              pseudo_samples_arr):
    datasets_with_pseudo_samples = {}
    for ps_num in pseudo_samples_arr:
        if ps_num == 0:
            datasets_with_pseudo_samples[0] = dataset_detected_outliers
            continue
        dataset = Transformer(method=oversampling_method).transform(dataset_detected_outliers, ps_num, detector,
                                                                    threshold)
        datasets_with_pseudo_samples[ps_num] = dataset
    return datasets_with_pseudo_samples


def add_noise_to_train_datasets(datasets, out_dims):
    for ps_num, dataset in datasets.items():
        tmp_dataset = add_noise_to_data(dataset, out_dims)
        if dataset.contains_pseudo_samples():
            tmp_dataset.set_pseudo_samples_indices_per_outlier(dataset.get_pseudo_sample_indices_per_outlier())
        datasets[ps_num] = tmp_dataset
    return datasets


def sort_files_by_dim(nav_files):
    nav_files_sort_by_dim = OrderedDict()
    for nfile in nav_files:
        data_dim = pd.read_csv(nfile[FileKeys.navigator_original_dataset_path]).shape[1]
        nav_files_sort_by_dim[data_dim] = nfile
    return dict(sorted(nav_files_sort_by_dim.items()))


def read_nav_files(path_to_dir, path_must_have_word=None):
    nav_files = []
    nav_files_paths = get_files_recursively(path_to_dir, FileNames.navigator_fname)
    tmp_nav_files = []
    for f in nav_files_paths:
        if path_must_have_word is not None and path_must_have_word in f:
            tmp_nav_files.append(f)
    if path_must_have_word is not None:
        nav_files_paths = tmp_nav_files
    for f in nav_files_paths:
        with open(f) as json_file:
            nav_files.append(json.load(json_file))
    return nav_files


def __convert_array_to_int_type(array):
    if array is None:
        return None
    return [int(x) for x in array]


def __convert_array_to_float_type(array):
    if array is None:
        return None
    return [float(x) for x in array]


def fs_key(fs):
    if fs is True:
        return "fs"
    else:
        return "no_fs"
