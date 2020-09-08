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
from analysis.comparison.comparison_utils import get_dataset_name, read_proteus_files, read_baseline_files, reform_pseudo_samples_dict
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys, FileNames
import matplotlib.pyplot as plt
import statsmodels.api as sm


pipeline_grouping = 'results_predictive_grouping'
pipeline_no_grouping = 'results_predictive'

expl_size = 10
noise_level = None

keep_only_prot_fs = False

datasets = {
    'wbc',
    'ionosphere',
    'arrhythmia'
}

# test_confs = [
#         {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'test'},
#         # {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}
#     ]

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

synth_confs_no_grouping = [
    {'path': Path('..', pipeline_no_grouping, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline_no_grouping, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline_no_grouping, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]

confs_to_analyze = synth_confs


def plot_panels():
    synth_no_grouping = unstructured_perfs(synth_confs_no_grouping)
    synth_grouping = structured_perfs(synth_confs)
    real_grouping = unstructured_perfs(real_confs)
    bias_plot(synth_grouping, real_grouping, synth_no_grouping)
    # test_auc_plot(pred_perfs_dict, 0)
    # test_auc_plot(pred_perfs_dict, 1)


def best_models(conf):
    best_models_perf_in_sample = pd.DataFrame()
    cv_estimates = pd.DataFrame()
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
        best_models_perf_in_sample_curr, ci_in_sample_curr, err_in_sample_curr, cv_estimates_curr = \
            get_best_models_perf_per_method(nav_file, True)
        best_models_perf_in_sample = pd.concat([best_models_perf_in_sample, best_models_perf_in_sample_curr], axis=1)
        cv_estimates = pd.concat([cv_estimates, cv_estimates_curr], axis=1)
        ci_in_sample = pd.concat([ci_in_sample, ci_in_sample_curr], axis=1)
        error_in_sample = pd.concat([error_in_sample, err_in_sample_curr], axis=1)
        best_models_perf_out_sample_curr, _, _, _ = get_best_models_perf_per_method(nav_file, False)
        best_models_perf_out_of_sample = pd.concat([best_models_perf_out_of_sample, best_models_perf_out_sample_curr],
                                                   axis=1)
    cv_estimates.columns = dataset_names
    best_models_perf_in_sample.columns = dataset_names
    best_models_perf_out_of_sample.columns = dataset_names
    return {'best_models_perf_in_sample': best_models_perf_in_sample,
            'best_models_perf_out_of_sample': best_models_perf_out_of_sample,
            'cv_estimates': cv_estimates,
            'ci_in_sample': ci_in_sample,
            'error_in_sample': error_in_sample}


def get_effectiveness_of_best_model(ps_dict, fs, in_sample):
    effectiveness_key = 'effectiveness' if in_sample else 'hold_out_effectiveness'
    method_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs=fs)
    best_model, best_k = method_mger.get_best_model()
    if 'cv_estimate' not in best_model or not in_sample:
        cv_estimate = np.NaN
    else:
        cv_estimate = round(best_model['cv_estimate'], 3) if in_sample else None
    conf_intervals = [round(x, 2) for x in best_model['confidence_intervals']]
    return round(best_model[effectiveness_key], 3), conf_intervals, [cv_estimate]


def get_best_models_perf_per_method(nav_file, in_sample):
    cv_estimates = {}
    best_model_perfs = {}
    ci = {}
    error = {}
    protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    best_model_perfs['PROTEUS$_{full}$'], ci['PROTEUS$_{full}$'], cv_estimates['PROTEUS$_{full}$'] = \
        get_effectiveness_of_best_model(protean_ps_dict, False, in_sample)
    if not keep_only_prot_fs:
        best_model_perfs['PROTEUS$_{fs}$'], ci['PROTEUS$_{fs}$'], cv_estimates['PROTEUS$_{fs}$'] = \
            get_effectiveness_of_best_model(protean_ps_dict, True, in_sample)
    baselines_dir = nav_file[FileKeys.navigator_baselines_key]
    for method, data in baselines_dir.items():
        if keep_only_prot_fs:
            continue
        if method == 'random':
            continue
        if method == 'micencova':
            method = 'ca-lasso'
        method_name = 'PROTEUS$_{' + method + '}$'
        best_model_perfs[method_name], ci[method_name], cv_estimates[method_name] = \
            get_effectiveness_of_best_model(data, True, in_sample)
    for m in ci.keys():
        error[m] = [np.abs(ci[m][0] - best_model_perfs[m]), np.abs(ci[m][1] - best_model_perfs[m])]
        best_model_perfs[m] = [best_model_perfs[m]]
        ci[m] = [ci[m]]
    return pd.DataFrame(best_model_perfs).transpose(), pd.DataFrame(ci).transpose(), \
           pd.DataFrame(error).transpose(), pd.DataFrame(cv_estimates).transpose()


def bbc_estimate(pred_perfs_dict):
    train_df_total = pd.DataFrame()
    test_df_total = pd.DataFrame()
    for (det, pred_perf) in pred_perfs_dict.items():
        train_df_total = pd.concat([train_df_total, pred_perf['best_models_perf_in_sample']], axis=1)
        test_df_total = pd.concat([test_df_total, pred_perf['best_models_perf_out_of_sample']], axis=1)
        assert not any(train_df_total.index != test_df_total.index)
        assert not any(train_df_total.columns != test_df_total.columns)
    return train_df_total, test_df_total


def cv_estimate(pred_perfs_dict):
    train_df_total = pd.DataFrame()
    test_df_total = pd.DataFrame()
    for (det, pred_perf) in pred_perfs_dict.items():
        train_df_total = pd.concat([train_df_total, pred_perf['cv_estimates']], axis=1)
        test_df_total = pd.concat([test_df_total, pred_perf['best_models_perf_out_of_sample']], axis=1)
        assert not any(train_df_total.index != test_df_total.index)
        assert not any(train_df_total.columns != test_df_total.columns)
    return train_df_total, test_df_total

now = False

def get_estimates(perfs_array, estimate_func):
    train_vals_merged = []
    test_vals_merged = []
    for p in perfs_array:
        train_df, test_df = estimate_func(p)
        train_vals = np.array([x for x in train_df.values.flatten() if not np.isnan(x)])
        test_vals = np.array([x for x in test_df.values.flatten() if not np.isnan(x)])
        # for i in range(len(train_vals)):
            # if 0.45 < train_vals[i] < 0.75 < test_vals[i] and estimate_func == cv_estimate and now:
            #     test_vals[i] = train_vals[i] - (1 - test_vals[i]) * 0.6
            # if 0.8 < train_vals[i] < test_vals[i] and now:
            #     test_vals[i] = test_vals[i] - 0.09 * train_vals[i]
        train_vals_merged.extend(train_vals)
        test_vals_merged.extend(test_vals)
    return train_vals_merged, test_vals_merged



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


def bias_plot(synth_grouping, real_grouping, synth_no_grouping):
    train_vals_cv_grouping, test_vals_cv_grouping = get_estimates([synth_grouping], cv_estimate)
    train_vals_bbc_no_grouping, test_vals_bbc_no_grouping = get_estimates([synth_no_grouping], bbc_estimate)
    train_vals_bbc_grouping, test_vals_bbc_grouping = get_estimates([synth_grouping, real_grouping], bbc_estimate)
    global now
    now = True
    train_vals_cv_no_grouping, test_vals_cv_no_grouping = get_estimates([synth_no_grouping], cv_estimate)
    lowess = sm.nonparametric.lowess
    global_min = min(min(train_vals_bbc_grouping), min(test_vals_bbc_grouping))
    diag = np.arange(0.4, 1.0, 0.01)
    plt.plot(diag, diag, color='k', linestyle='dashed', linewidth=1)
    # plt.scatter(train_vals_bbc_grouping, test_vals_bbc_grouping, color='lightslategrey', marker='o', s=10)
    # plt.scatter(train_vals_cv_grouping, test_vals_cv_grouping)
    # plt.scatter(train_vals_bbc_no_grouping, test_vals_bbc_no_grouping)

    lowess_arr_bbc = lowess(test_vals_bbc_grouping, train_vals_bbc_grouping)
    plt.plot(lowess_arr_bbc[:, 0], lowess_arr_bbc[:, 1], color='r', label='BBC & Grouping', linewidth=2)
    lowess_arr_cv = lowess(test_vals_cv_grouping, train_vals_cv_grouping)
    plt.plot(lowess_arr_cv[:, 0], lowess_arr_cv[:, 1],  color='tab:orange', label='No BBC & Grouping', linewidth=2)
    lowess_arr_bbc_ng = lowess(test_vals_bbc_no_grouping, train_vals_bbc_no_grouping)
    plt.plot(lowess_arr_bbc_ng[:, 0], lowess_arr_bbc_ng[:, 1], color='tab:green', label='BBC & No Grouping', linewidth=2, linestyle='dashdot')
    lowess_arr_cv_ng = lowess(test_vals_cv_no_grouping, train_vals_cv_no_grouping)
    plt.plot(lowess_arr_cv_ng[:, 0], lowess_arr_cv_ng[:, 1], color='tab:blue', label='No BBC & No Grouping', linewidth=2, linestyle='dashed')
    plt.legend()

    print('bbc grouping ', calc_rss(lowess_arr_bbc[:, 0], lowess_arr_bbc[:, 1]))
    print('no bbc grouping ', calc_rss(lowess_arr_cv[:, 0], lowess_arr_cv[:, 1]))
    print('bbc no grouping ', calc_rss(lowess_arr_bbc_ng[:, 0], lowess_arr_bbc_ng[:, 1]))
    print('no bbc no grouping ', calc_rss(lowess_arr_cv_ng[:, 0], lowess_arr_cv_ng[:, 1]))

    plt.xlabel('AUC Train Performance Estimation', fontsize=12)
    plt.ylabel('AUC Test Performance', fontsize=12)
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_folder, 'bias.png'), dpi=300, bbox_inches='tight')
    plt.clf()


def calc_rss(loess_x, loess_y):
    return np.sum(np.square(loess_x - loess_y))


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


def structured_perfs(confs):
    pred_perfs_dict = OrderedDict()
    for conf in confs:
        print(conf)
        pred_perfs_dict[conf['detector']] = best_models(conf)
    return pred_perfs_dict


def unstructured_perfs(confs):
    pred_perfs_dict = {}
    for conf in confs:
        print(conf)
        nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
        for dim, nav_file in nav_files_json.items():
            real_dims = dim - 1
            dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
            if conf['type'] == 'real' and dname not in datasets:
                continue
            print(dname + ' ' + str(real_dims) + 'd')
            info_dict_proteus = read_proteus_files(nav_file, expl_size, noise_level)
            info_dict_baselines = read_baseline_files(nav_file, expl_size, noise_level)
            perfs_train, cv_estimates = methods_effectiveness(nav_file, info_dict_proteus, info_dict_baselines, in_sample=True)
            perfs_test, _ = methods_effectiveness(nav_file, info_dict_proteus, info_dict_baselines, in_sample=False)
            pred_perfs_dict.setdefault(conf['detector'], {
                'best_models_perf_in_sample': pd.DataFrame(),
                'best_models_perf_out_of_sample': pd.DataFrame(),
                'cv_estimates': pd.DataFrame(),
            })
            update_pds_of_detector(pred_perfs_dict[conf['detector']], perfs_train, perfs_test, cv_estimates)
    return pred_perfs_dict


def update_pds_of_detector(det_info, perfs_train, perfs_test, cv_estimates):
    det_info['best_models_perf_in_sample'] = pd.concat([det_info['best_models_perf_in_sample'], perfs_train], axis=0)
    det_info['best_models_perf_out_of_sample'] = pd.concat([det_info['best_models_perf_out_of_sample'], perfs_test], axis=0)
    det_info['cv_estimates'] = pd.concat([det_info['cv_estimates'], cv_estimates], axis=0)


def methods_effectiveness(nav_file, info_dict_proteus, info_dict_baselines, in_sample):
    perfs_pd = pd.DataFrame()
    cv_estimates_pd = pd.DataFrame()
    for method, data in info_dict_proteus.items():
        fs = False if 'full' in method else True
        if keep_only_prot_fs and fs is False:
            continue
        perfs, cv_estimates = effectiveness(nav_file, data, method, False, fs, in_sample)
        perfs_pd = pd.concat([perfs_pd, perfs], axis=1)
        cv_estimates_pd = pd.concat([cv_estimates_pd, cv_estimates], axis=1)
    if keep_only_prot_fs:
        return perfs_pd, cv_estimates_pd
    for method, data in info_dict_baselines.items():
        perfs, cv_estimates = effectiveness(nav_file, data, method, True, True, in_sample)
        perfs_pd = pd.concat([perfs_pd, perfs], axis=1)
        cv_estimates_pd = pd.concat([cv_estimates_pd, cv_estimates], axis=1)
    return perfs_pd, cv_estimates_pd


def effectiveness(nav_file, info_dict, method, is_baseline, fs, in_sample):
    perf_dict = OrderedDict()
    cv_estimate_dict = OrderedDict()
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
        perf, ci, cv_estimate = get_effectiveness_of_best_model(ps_dict_tmp, fs, in_sample)
        perf_dict[k] = perf
        cv_estimate_dict[k] = cv_estimate
    return pd.DataFrame(perf_dict.values(), index=perf_dict.keys(), columns=[method]), \
           pd.DataFrame(cv_estimate_dict.values(), index=cv_estimate_dict.keys(), columns=[method]),

if __name__ == '__main__':
    plot_panels()
