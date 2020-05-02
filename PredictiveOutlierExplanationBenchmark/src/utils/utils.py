import json
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


def fs_key(fs):
    if fs is True:
        return "fs"
    else:
        return "no_fs"
