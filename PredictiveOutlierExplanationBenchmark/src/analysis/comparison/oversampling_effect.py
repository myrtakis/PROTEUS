from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandpadir)
from collections import OrderedDict
from utils.helper_functions import read_nav_files, sort_files_by_dim
from analysis.comparison.comparison_utils import get_dataset_name
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys, FileNames
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json

expl_size = 10
noise_level = 0

pipeline = 'results_predictive'

datasets = {
    'wbc': 'Breast Cancer',
    'arrhythmia': 'Arrhythmia',
    'ionosphere': 'Ionosphere'
}


test_confs = [
        #{'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'},
        {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}
    ]

real_confs =[
    {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'},
    {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'},
    {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}
]


def analyze_oversampling_effect():
    test_perfs_dataset = {}
    for conf in real_confs:
        print(conf)
        nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
        for dim, nav_file in nav_files_json.items():
            real_dims = dim - 1
            dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
            if dname not in datasets:
                continue
            print(dname + ' ' + str(real_dims) + 'd')
            info_dict_proteus = read_proteus_files(nav_file)
            perfs_test = methods_effectiveness(nav_file, info_dict_proteus, True, conf['detector'])
            if datasets[dname] not in test_perfs_dataset:
                test_perfs_dataset[datasets[dname]] = perfs_test['fs'][noise_level]
            else:
                test_perfs_dataset[datasets[dname]] = pd.concat([test_perfs_dataset[datasets[dname]],
                                                                 perfs_test['fs'][noise_level]], axis=1)
    print()


def read_proteus_files(nav_file):
    info_files_dict = {}
    for ps_samples, data in nav_file[FileKeys.navigator_pseudo_samples_key].items():
        ps_dir = Path(data[FileKeys.navigator_pseudo_sample_dir_key]).parent
        assert str(ps_dir).endswith(ps_samples)
        update_info_dict(info_files_dict, ps_dir, ps_samples)
    return {'full': sort_info_dict(info_files_dict),
            'fs': sort_info_dict(info_files_dict)}


def read_baseline_files(nav_file):
    info_files_dict = {}
    for method, method_data in nav_file[FileKeys.navigator_baselines_key].items():
        if method == 'random':
            continue
        info_files_dict[method] = {}
        for ps_samples, ps_data in method_data.items():
            update_info_dict(info_files_dict[method], ps_data[FileKeys.navigator_pseudo_sample_dir_key], ps_samples)
        info_files_dict[method] = sort_info_dict(info_files_dict[method])
    return info_files_dict


def update_info_dict(info_file_dict, folder, ps_samples):
    for r, d, f in os.walk(folder):
        for file in f:
            if FileNames.info_file_fname == file:
                info_dict = read_info_file(Path(r, file))
                if valid_info_file(info_dict):
                    if expl_size is None:
                        key = info_dict[FileKeys.info_file_explanation_size]
                    else:
                        key = info_dict[FileKeys.info_file_noise_level]
                    info_file_dict.setdefault(key, {})
                    info_file_dict[key][ps_samples] = r


def valid_info_file(info_dict):
    assert expl_size is not None or noise_level is not None
    if expl_size is not None and info_dict[FileKeys.info_file_explanation_size] != expl_size:
        return False
    if noise_level is not None and info_dict[FileKeys.info_file_noise_level] != noise_level:
        return False
    return True


def sort_info_dict(info_dict):
    info_dict_sorted = OrderedDict()
    for k in sorted(info_dict.keys()):
        info_dict_sorted[k] = info_dict[k]
    return info_dict_sorted


def read_info_file(info_file):
    with open(info_file) as json_file:
        info_dict = json.load(json_file)
        return info_dict


def methods_effectiveness(nav_file, info_dict_proteus, in_sample, detector):
    method_perfs_pd = OrderedDict()
    for method, data in info_dict_proteus.items():
        if 'full' in method:
            continue
        method_perfs_pd[method] = effectiveness(nav_file, data, method, False, True, in_sample, detector)
    return method_perfs_pd


def effectiveness(nav_file, info_dict, method, is_baseline, fs, in_sample, detector):
    perf_dict = OrderedDict()
    if is_baseline:
        ps_dict = nav_file[FileKeys.navigator_baselines_key]
    else:
        ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    for k, v in info_dict.items():
        for ps_key, dir in v.items():
            if is_baseline:
                reform_pseudo_samples_dict(ps_dict[method][ps_key], dir)
            else:
                reform_pseudo_samples_dict(ps_dict[ps_key], dir)
        if is_baseline:
            ps_mger = PseudoSamplesMger(ps_dict[method], 'roc_auc', fs)
        else:
            ps_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs)
        perfs_per_k = get_effectiveness_of_best_model(ps_mger, in_sample)
        detector = 'iForest' if detector == 'iforest' else detector.upper()
        perf_dict[k] = pd.DataFrame(perfs_per_k.values(), index=perfs_per_k.keys(), columns=[detector])
    return perf_dict


def get_effectiveness_of_best_model(ps_mger, in_sample):
    effectiveness_key = 'effectiveness' if in_sample else 'hold_out_effectiveness'
    best_model_per_k = ps_mger.get_best_model_per_k()
    perfs_per_k = OrderedDict()
    for k, data in sorted(best_model_per_k.items()):
        perfs_per_k[k] = data[effectiveness_key]
    return perfs_per_k


def reform_pseudo_samples_dict(ps_dict, dir):
    old_dir = ps_dict[FileKeys.navigator_pseudo_sample_dir_key]
    for k, v in ps_dict.items():
        if type(v) is str and old_dir in v:
            ps_dict[k] = old_dir.replace(old_dir, dir)
    return ps_dict


if __name__ == '__main__':
    analyze_oversampling_effect()