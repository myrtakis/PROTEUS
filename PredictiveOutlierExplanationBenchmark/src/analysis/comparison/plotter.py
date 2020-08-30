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


pipeline = 'results_predictive'

test_confs = [
        #{'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'},
        {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}
    ]

synth_confs =[
    {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]

confs_to_analyze = synth_confs


def plot_panels():
    pred_perfs_dict = OrderedDict()
    for conf in confs_to_analyze:
        print(conf)
        pred_perfs_dict[conf['detector']] = best_models(conf)
    bias_plot(pred_perfs_dict)
    test_auc_plot(pred_perfs_dict, 0)
    test_auc_plot(pred_perfs_dict, 1)


def best_models(conf):
    best_models_perf_in_sample = pd.DataFrame()
    ci_in_sample = pd.DataFrame()
    error_in_sample = pd.DataFrame()
    best_models_perf_out_of_sample = pd.DataFrame()
    dataset_names = []
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
        print(dname + ' ' + str(real_dims) + 'd')
        rel_fratio = '(' + str(int(round((dim-5)/dim, 2) * 100)) + '%)' if conf['type'] != 'real' else ''
        dataset_names.append(dname + ' ' + str(real_dims) + 'd ' + rel_fratio)
        # time_df = pd.concat([time_df, get_time_per_method(nav_file)], axis=1)
        best_models_perf_in_sample_curr, ci_in_sample_curr, err_in_sample_curr = get_best_models_perf_per_method(
            nav_file, True)
        best_models_perf_in_sample = pd.concat([best_models_perf_in_sample, best_models_perf_in_sample_curr], axis=1)
        ci_in_sample = pd.concat([ci_in_sample, ci_in_sample_curr], axis=1)
        error_in_sample = pd.concat([error_in_sample, err_in_sample_curr], axis=1)
        best_models_perf_out_sample_curr, _, _ = get_best_models_perf_per_method(nav_file, False)
        best_models_perf_out_of_sample = pd.concat([best_models_perf_out_of_sample, best_models_perf_out_sample_curr],
                                                   axis=1)
    best_models_perf_in_sample.columns = dataset_names
    best_models_perf_out_of_sample.columns = dataset_names
    return {'best_models_perf_in_sample': best_models_perf_in_sample,
            'best_models_perf_out_of_sample': best_models_perf_out_of_sample,
            'ci_in_sample': ci_in_sample,
            'error_in_sample': error_in_sample}


def get_effectiveness_of_best_model(ps_dict, fs, in_sample):
    effectiveness_key = 'effectiveness' if in_sample else 'hold_out_effectiveness'
    method_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs=fs)
    best_model, best_k = method_mger.get_best_model()
    conf_intervals = [round(x, 2) for x in best_model['confidence_intervals']]
    return round(best_model[effectiveness_key], 3), conf_intervals


def get_best_models_perf_per_method(nav_file, in_sample):
    best_model_perfs = {}
    ci = {}
    error = {}
    protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    best_model_perfs['PROTEUS$_{full}$'], ci['PROTEUS$_{full}$'] = get_effectiveness_of_best_model(protean_ps_dict, False,
                                                                                               in_sample)
    best_model_perfs['PROTEUS$_{fs}$'], ci['PROTEUS$_{fs}$'] = get_effectiveness_of_best_model(protean_ps_dict, True,
                                                                                           in_sample)
    baselines_dir = nav_file[FileKeys.navigator_baselines_key]
    for method, data in baselines_dir.items():
        if method == 'random':
            continue
        if method == 'micencova':
            method = 'ca-lasso'
        method_name = 'PROTEUS$_{' + method + '}$'
        best_model_perfs[method_name], ci[method_name] = get_effectiveness_of_best_model(data, True, in_sample)
    for m in ci.keys():
        error[m] = [np.abs(ci[m][0] - best_model_perfs[m]), np.abs(ci[m][1] - best_model_perfs[m])]
        best_model_perfs[m] = [best_model_perfs[m]]
        ci[m] = [ci[m]]
    return pd.DataFrame(best_model_perfs).transpose(), pd.DataFrame(ci).transpose(), pd.DataFrame(error).transpose()


def calculate_bias(pred_perfs_dict):
    train_df_total = pd.DataFrame()
    test_df_total = pd.DataFrame()
    for (det, pred_perf) in pred_perfs_dict.items():
        train_df_total = pd.concat([train_df_total, pred_perf['best_models_perf_in_sample']], axis=1)
        test_df_total = pd.concat([test_df_total, pred_perf['best_models_perf_out_of_sample']], axis=1)
        assert not any(train_df_total.index != test_df_total.index)
        assert not any(train_df_total.columns != test_df_total.columns)
    return train_df_total, test_df_total


def calculate_error(pred_perfs_dict):
    error_df = pd.DataFrame()
    for (det, pred_perf) in pred_perfs_dict.items():
        error_df = pd.concat([error_df,  pred_perf['error_in_sample']], axis=1)
    lb_error = [x for x in error_df.iloc[:, 0::2].values.flatten() if not np.isnan(x)]
    ub_error = [x for x in error_df.iloc[:, 1::2].values.flatten() if not np.isnan(x)]
    return lb_error, ub_error


def average_out_dim(pred_perfs_dict, option):
    average_df = pd.DataFrame()
    fs_methods = 5
    for (det, pred_perf) in pred_perfs_dict.items():
        test_df = pred_perf['best_models_perf_out_of_sample']
        if test_df.shape[0] < fs_methods:
            loda = pd.DataFrame(OrderedDict.fromkeys(test_df.columns, 1), index=['PROTEUS$_{loda}$'])
            test_df = pd.concat([test_df, loda], axis=0)
        if option == 0:
            if len(average_df) == 0:
                average_df = test_df
            else:
                average_df += test_df
        else:
            det = det.upper() if det != 'iforest' else 'iForest'
            dimensions_avg = pd.DataFrame(test_df.mean(axis=1), columns=[det])
            if det != 'LODA':
                dimensions_avg.iloc[-1, -1] = np.NaN
            average_df = pd.concat([average_df, dimensions_avg], axis=1)
    if option == 0:
        average_df /= len(pred_perfs_dict)
    return average_df


def bias_plot(pred_perfs_dict):
    train_df, test_df = calculate_bias(pred_perfs_dict)
    lowess = sm.nonparametric.lowess
    lb_error, ub_error = calculate_error(pred_perfs_dict)
    train_vals = np.array([x for x in train_df.values.flatten() if not np.isnan(x)])
    test_vals = np.array([x for x in test_df.values.flatten() if not np.isnan(x)])
    plt.plot([min(train_vals), max(train_vals)], [min(test_vals), max(test_vals)])
    plt.scatter(train_vals, test_vals, color='r', marker='o')
    # plt.fill_between(x, x - lb_error, x + ub_error, alpha=0.2)
    lowess_arr = lowess(test_vals, train_vals)
    plt.plot(lowess_arr[:, 0], lowess_arr[:, 1], color='tab:green')
    plt.xlabel('AUC Train Performance Estimation', fontsize=14)
    plt.ylabel('AUC Test Performance', fontsize=14)
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_folder, 'bias.png'), dpi=300, bbox_inches='tight')
    plt.clf()


def test_auc_plot(pred_perfs_dict, option):
    leg_handles_dict = {
        'PROTEUS$_{full}$': 'tab:blue',
        'PROTEUS$_{fs}$': 'tab:orange',
        'PROTEUS$_{ca-lasso}$': 'tab:green',
        'PROTEUS$_{shap}$': 'tab:red',
        'PROTEUS$_{loda}$': 'tab:purple',
        'PROTEUS$_{random}$': 'cyan',
    }
    avg_df = average_out_dim(pred_perfs_dict, option)
    colors = []
    for k in avg_df.index:
        colors.append(leg_handles_dict[k])
    markers = ["-s", "-o", "-v", "-^", "-*"]
    avg_df.transpose().plot(style=markers, color=colors)
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(np.arange(avg_df.shape[1]), avg_df.columns, fontsize=11)
    plt.ylim([0, 1.1])
    plt.ylabel('Test AUC')
    xlabel = 'Data Dimensionality (Noisy Features Ratio)' if option is 0 else 'Detectors'
    plt.xlabel(xlabel, fontsize=13, labelpad=5)
    plt.legend(prop={'size': 12})
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    if option == 0:
        fname = 'synth-dim-test-auc.png'
    else:
        fname = 'synth-det-test-auc.png'
    plt.savefig(Path(output_folder, fname), dpi=300, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plot_panels()
