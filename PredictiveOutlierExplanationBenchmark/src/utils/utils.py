import json
from pathlib import Path

from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
import pandas as pd
import os


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
        optimal_features = optimal_features.union([int(f) for f in s[s.index('[')+1: s.index(']')].split()])
    return optimal_features


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


def get_best_model_features_original_data(original_data_results_path, metric_id):
    best_model_orig = Path(original_data_results_path, FileNames.best_model_fname)
    with open(best_model_orig) as json_file:
        best_model = json.load(json_file)[fs_key(True)][metric_id]
        fsel_id = best_model['feature_selection']['id']
        features = best_model['feature_selection']['features']
        clf_id = best_model['classifiers']['id']
        best_conf = fsel_id + '_' + clf_id
        return features, best_conf


def fs_key(fs):
    if fs is True:
        return "fs"
    else:
        return "no_fs"
