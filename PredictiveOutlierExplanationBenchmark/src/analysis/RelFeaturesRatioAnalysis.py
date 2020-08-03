import os, inspect, sys
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)

from utils import helper_functions
from configpkg import ConfigMger, DatasetConfig
from holders.Dataset import Dataset
from utils.helper_functions import sort_files_by_dim, read_nav_files
from utils.shared_names import *
from comparison.comparison_utils import get_dataset_name
from utils.pseudo_samples import PseudoSamplesMger
import pandas as pd
import json
import numpy as np
from pipeline.automl.automl_constants import MAX_FEATURES
import matplotlib.pyplot as plt

pipeline = 'results_predictive'

test_confs = [
    {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}
]
# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}

synth_confs =[
    {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]


# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}

confs_to_analyze = synth_confs

def analyze():
    fig_name = None
    prec_dict = {}
    recall_dict = {}
    for conf in confs_to_analyze:
        if fig_name is None:
            fig_name = 'real_features_perf.png' if conf['type'] == 'real' else 'synthetic_features_perf.png'
        nav_files = read_nav_files(conf['path'], conf['type'])
        nav_files = sort_files_by_dim(nav_files)
        feature_perf_prec, feature_perf_recall = analysis_per_nav_file(nav_files, conf)
        prec_dict[conf['detector']] = feature_perf_prec
        recall_dict[conf['detector']] = feature_perf_recall

    # tmp_dict = {'lof': None, 'iforest': None, 'loda': None}
    # indexes = ['PROTEUS_{fs}', 'PROTEUS_{ca-lasso}', 'PROTEUS_{shap}', 'PROTEUS_{loda}']
    # columns = ['S 20d (10%)', 'S 20d (10%)', 'S 20d (10%)', 'S 20d (10%)', 'S 100d (10%)']
    # for k in tmp_dict.keys():
    #     tmp_dict[k] = pd.DataFrame(np.random.rand(4, 5), index=indexes, columns=columns)
    # plot_dataframes(tmp_dict, tmp_dict, fig_name)

    plot_dataframes(prec_dict, recall_dict, fig_name)


def analysis_per_nav_file(nav_files, conf):
    dataset_names = []
    feature_perf_prec = pd.DataFrame()
    feature_perf_recall = pd.DataFrame()
    for dim, nav_file in nav_files.items():
        ConfigMger.setup_configs(nav_file[FileKeys.navigator_conf_path])
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] == 'synthetic')
        print(dname + ' ' + str(real_dims) + '-d')
        dataset_names.append(dname + ' ' + str(real_dims) + '-d')
        methods_features = get_selected_features_per_method(nav_file)
        rel_features = get_relevant_features(nav_file, conf)
        recall, prec = calculate_feature_metrics(methods_features, rel_features)
        feature_perf_prec = pd.concat([feature_perf_prec, prec], axis=1)
        feature_perf_recall = pd.concat([feature_perf_recall, recall], axis=1)
    feature_perf_prec.columns = dataset_names
    feature_perf_recall.columns = dataset_names
    return feature_perf_prec, feature_perf_recall


def get_selected_features_per_method(nav_file):
    methods_sel_features = {}
    protean_psmger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], 'roc_auc', fs=True)
    best_model, best_k = protean_psmger.get_best_model()
    methods_sel_features['PROTEUS_{fs}'] = best_model['feature_selection']['features']
    methods_explanations_file = Path(nav_file[FileKeys.navigator_baselines_dir_key], FileNames.baselines_explanations_fname)
    with open(methods_explanations_file) as json_file:
        explanations = json.load(json_file)
        for method, data in explanations.items():
            if method == 'random':
                continue
            if method == 'micencova':
                method = 'ca-lasso'
            features_sorted = np.argsort(np.array(data['global_explanation']))[::-1]
            method_name = 'PROTEUS_{' + method + '}'
            methods_sel_features[method_name] = features_sorted[:MAX_FEATURES]
    return methods_sel_features


def get_relevant_features(nav_file, conf):
    orig_data = Dataset(nav_file[FileKeys.navigator_original_dataset_path], DatasetConfig.get_anomaly_column_name(),
                        DatasetConfig.get_subspace_column_name())
    rel_features = orig_data.get_relevant_features()
    if len(rel_features) == 0:
        assert conf['type'] == 'real'
        rel_features = set(np.arange(orig_data.get_X().shape[1]))
    return rel_features


def calculate_feature_metrics(methods_features, rel_features):
    methods_precision = {}
    methods_recall = {}
    for method, sel_features in methods_features.items():
        methods_precision[method] = features_precision(sel_features, rel_features)
        methods_recall[method] = features_recall(sel_features, rel_features)
    methods_recall_as_df = pd.DataFrame(methods_recall.values(), index=methods_recall.keys())
    methods_precision_as_df = pd.DataFrame(methods_precision.values(), index=methods_precision.keys())
    return methods_recall_as_df, methods_precision_as_df


def plot_dataframes(prec_perfs, recall_perfs, fig_name):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 4), sharex=True)
    leg_handles_dict = {
        'PROTEUS_{full}': ('tab:blue', '$PROTEUS_{full}$'),
        'PROTEUS_{fs}': ('tab:orange', '$PROTEUS_{fs}$'),
        'PROTEUS_{ca-lasso}': ('tab:green', '$PROTEUS_{ca-lasso}$'),
        'PROTEUS_{shap}': ('tab:red', '$PROTEUS_{shap}$'),
        'PROTEUS_{loda}': ('tab:purple', '$PROTEUS_{loda}$'),
        'PROTEUS_{random}': ('cyan', '$PROTEUS_{random}$')
    }
    leg_handles_arr = []
    colors = []
    markers = ["-s", "-o", "-v", "-^"]
    keys = list(prec_perfs.keys())
    loda_i = loda_j = 0
    for m in prec_perfs[keys[0]].index:
        leg_handles_arr.append(leg_handles_dict[m][1])
        colors.append(leg_handles_dict[m][0])
    for i, perf_dict in enumerate([prec_perfs, recall_perfs]):
        for j, (det, perf_df) in enumerate(perf_dict.items()):
            det = det.upper()
            if det == 'IFOREST':
                det = 'iForest'
            axes[i, j].locator_params(axis='x', nbins=perf_df.shape[0])
            axes[i, j].set_ylim((0, 1.05))
            axes[i, j].set_yticks(np.arange(0, 1.2, 0.2))
            axes[i, j].set_title(det, fontsize=13)
            ytitle = 'Precision' if i == 1 else 'Recall'
            axes[i, j].set_ylabel(ytitle, fontsize=13)
            axes[i, j].set_xticklabels(labels=list(perf_df.index), ha='center')
            if det == 'LODA':
                loda_i = i
                loda_j = j
            perf_df.transpose().plot(ax=axes[i, j], legend=None, rot=25, style=markers, color=colors)
    handles, labels = axes[loda_i, loda_j].get_legend_handles_labels()
    fig.legend(handles, leg_handles_arr, loc='upper center', ncol=4, fontsize=14)
    fig.subplots_adjust(wspace=0.6, hspace=0.5, bottom=0.2, top=0.8)
    # plt.tight_layout()
    x0, x1, y0, y1 = plt.axis()
    margin_x = 0.1 * (x1 - x0)
    margin_y = 0.1 * (y1 - y0)
    plt.axis((x0 - margin_x, x1 + margin_x, y0 - margin_y, y1 + margin_y))
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_folder, fig_name), dpi=300)
    plt.clf()


def features_precision(selected_features, optimal_features):
    return len(optimal_features.intersection(selected_features)) / len(selected_features)


def features_recall(selected_features, optimal_features):
    return len(optimal_features.intersection(selected_features)) / len(optimal_features)


if __name__ == '__main__':
    analyze()