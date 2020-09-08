from pathlib import Path
from utils.helper_functions import read_nav_files, sort_files_by_dim
from analysis.comparison.comparison_utils import get_dataset_name, read_proteus_files, read_baseline_files, reform_pseudo_samples_dict
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys, FileNames
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import json
import glob
from models.Classifier import Classifier


pipeline_grouping = 'results_predictive_grouping'
pipeline_no_grouping = 'results_predictive'

expl_size=10
noise_level = None

datasets = {
    'wbc',
    'ionosphere',
    'arrhythmia'
}


synth_confs =[
    {'path': Path('..', pipeline_grouping, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline_grouping, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline_grouping, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]

real_confs = [
    {'path': Path('..', pipeline_grouping, 'iforest'), 'detector': 'iforest', 'type': 'real'},
    {'path': Path('..', pipeline_grouping, 'lof'), 'detector': 'lof', 'type': 'real'},
    {'path': Path('..', pipeline_grouping, 'loda'), 'detector': 'loda', 'type': 'real'}
]


synth_confs_no_grouping =[
    {'path': Path('..', pipeline_no_grouping, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline_no_grouping, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline_no_grouping, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]


def calc_estimates_synth_data():
    for conf in synth_confs:
        print(conf)
        best_models_unstructured(conf)


def calc_estimates_real_data():
    for conf in real_confs:
        print(conf)
        best_models_unstructured(conf)


def calc_estimates_synth_data_no_grouping():
    for conf in synth_confs_no_grouping:
        print(conf)
        best_models_unstructured(conf)


def best_models_unstructured(conf):
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    for dim, nav_file in nav_files_json.items():
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
        if conf['type'] == 'real' and dname not in datasets:
            continue
        info_dict_proteus = read_proteus_files(nav_file, expl_size, noise_level)
        info_dict_baselines = read_baseline_files(nav_file, expl_size, noise_level)
        for method, data in info_dict_proteus.items():
            fs = False if 'full' in method else True
            update_unstructure(nav_file, method, data, False, fs)
        for method, data in info_dict_baselines.items():
            update_unstructure(nav_file, method, data, True, True)


def update_unstructure(nav_file, method, info_dict, is_baseline, fs):
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
        ps_dict_tmp = ps_dict[method] if is_baseline else ps_dict
        update_best_model_cv_estimates(method, nav_file, ps_dict_tmp, fs)


def best_models_synth(conf):
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    for dim, nav_file in nav_files_json.items():
        protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
        update_best_model_cv_estimates('Full Selector', nav_file, protean_ps_dict, False)
        update_best_model_cv_estimates('FS', nav_file, protean_ps_dict, True)
        baselines_dir = nav_file[FileKeys.navigator_baselines_key]
        for method, data in baselines_dir.items():
            update_best_model_cv_estimates(method, nav_file, data, True)


def update_best_model_cv_estimates(method, nav_file, ps_dict, fs):
    method_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs=fs)
    best_model, best_k = method_mger.get_best_model()
    best_model_dir = method_mger.get_path_of_best_model()
    fs_key = 'fs' if fs else 'no_fs'
    if fs:
        best_model_updated = get_dict_cv_performance_updated(best_model, best_model_dir)
    else:
        best_model_updated = compute_cv_performance(nav_file, best_model_dir, best_model)
    write_update(best_model_dir, fs_key, best_model_updated)
    print('Cv estimate updated for method', method)


def write_update(best_model_dir, fs_key, best_model_updated):
    with open(Path(best_model_dir, FileNames.best_model_fname), 'r+', encoding='utf-8') as f:
        best_model_dict_all = json.load(f)
        best_model_dict_all[fs_key]['roc_auc'] = best_model_updated
    with open(Path(best_model_dir, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
        f.write(json.dumps(best_model_dict_all, indent=4, separators=(',', ': '), ensure_ascii=False))


def compute_cv_performance(nav_file, best_k_path, best_model):
    folds_inds = get_folds_inds(nav_file, best_k_path)
    dataset_path = find_dataset_path(best_k_path)
    df = pd.read_csv(dataset_path)
    if 'subspaces' in df.columns:
        df = df.drop(columns=['subspaces'])
    X = df.drop(columns=['is_anomaly'])
    Y = df['is_anomaly']
    best_model['cv_estimate'] = cv_perf_full_selector(X, Y, folds_inds, best_model)
    return best_model


def find_dataset_path(curr_dir):
    files = os.listdir(str(curr_dir))
    while not any([f.endswith('.csv') for f in files]):
        curr_dir = Path(curr_dir).parent
        files = os.listdir(str(curr_dir))
    for f in files:
        if f.endswith('.csv') and 'hold_out' not in f:
            return Path(curr_dir, f)


def get_dict_cv_performance_updated(best_model, path):
    final_dir = Path(path, FileNames.configurations_folder, FileNames.configurations_perfs_folder)
    fcounter = 1
    max_rep_fname, max_rep = None, None
    for f in os.listdir(final_dir):
        if f.endswith('.json'):
            rep_num = int(f[10: f.index('.')])
            if max_rep is None or max_rep < rep_num:
                max_rep = rep_num
                max_rep_fname = f
            fcounter += 1
    with open(Path(final_dir, max_rep_fname), 'r', encoding='utf-8') as json_file:
        cv_estimate = json.load(json_file)['roc_auc'][str(best_model['id'])] / fcounter
        best_model['cv_estimate'] = cv_estimate
    return best_model


def cv_perf_full_selector(X, Y, folds_inds, best_model):
    Y = np.array(Y)
    auc_in_reps = 0.
    for rep, folds in folds_inds.items():
        predictions_in_rep = []
        Y_ordered = []
        for k_fold, k_inds in folds.items():
            clf_conf = best_model['classifiers']
            print('\rRep', rep, ' / Fold', k_fold, ' /', clf_conf['id'], clf_conf['params'], end='')
            X_train, Y_train = X.iloc[k_inds['train_indices']], Y[k_inds['train_indices']]
            X_test, Y_test = X.iloc[k_inds['test_indices']], Y[k_inds['test_indices']]
            Y_ordered.extend(Y_test.tolist())
            clf = Classifier()
            clf.setup_classifier_manually(clf_conf['id'], clf_conf['params'])
            predictions_in_rep.extend(clf.train(X_train, Y_train).predict_proba(X_test))
        auc_in_reps += roc_auc_score(Y_ordered, predictions_in_rep)
    print()
    return auc_in_reps / len(folds_inds)


def get_folds_inds(nav_file, best_k_path):
    baseline_dir = Path(nav_file[FileKeys.navigator_baselines_dir_key])
    proteus_dir = Path(nav_file[FileKeys.navigator_train_hold_out_inds]).parent
    ps_inds_info = None
    inds_folder = None
    ps_inds_info_path = Path(best_k_path, FileNames.pseudo_samples_info)
    if ps_inds_info_path.exists():
        with open(ps_inds_info_path, 'r') as json_file:
            ps_inds_info = json.load(json_file)
    for path in [proteus_dir, baseline_dir]:
        tmp = next(filter(lambda x: FileNames.indices_folder in x, glob.glob(str(Path(path, '**')), recursive=True)), None)
        if tmp is not None:
            inds_folder = tmp
    folds_inds = {}
    for i, f in enumerate(os.listdir(inds_folder)):
        if not f.endswith('.json'):
            continue
        with open(Path(inds_folder, f), 'r') as json_file:
            folds_inds[i] = json.load(json_file)
    folds_inds_grouped = apply_grouping(folds_inds, ps_inds_info)
    return folds_inds_grouped


def apply_grouping(folds_inds, ps_info_dict):
    if ps_info_dict is None:
        return folds_inds
    for rep, folds in folds_inds.items():
        for k_fold, k_inds in folds.items():
            for anomaly_id, anom_ps_samples in ps_info_dict.items():
                if int(anomaly_id) in k_inds['train_indices']:
                    k_inds['train_indices'].extend(anom_ps_samples)
                elif int(anomaly_id) in k_inds['test_indices']:
                    k_inds['test_indices'].extend(anom_ps_samples)
    return folds_inds


if __name__ == '__main__':
    # calc_estimates_synth_data()
    # calc_estimates_real_data()
    calc_estimates_synth_data_no_grouping()
