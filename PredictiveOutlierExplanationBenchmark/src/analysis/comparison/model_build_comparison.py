from pathlib import Path
from matplotlib.font_manager import FontProperties
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandpadir)
from models.OutlierDetector import Detector
from baselines.posthoc_explanation_methods import ExplanationMethods
from configpkg import ConfigMger, DatasetConfig
from holders.Dataset import Dataset
from pipeline.automl.automl_processor import AutoML
from utils import metrics
from utils.helper_functions import read_nav_files, sort_files_by_dim
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys, FileNames
from analysis.comparison.comparison_utils import load_baseline_explanations, get_dataset_name
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

MAX_FEATURES = 10
pipeline = 'results_predictive'
holdout = True
build_models = False  # compare the built models

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}


# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'synthetic'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'}
conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}


def compare_models():
    print(conf)
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    best_models_perf_in_sample = pd.DataFrame()
    ci_in_sample = pd.DataFrame()
    error_in_sample = pd.DataFrame()
    best_models_perf_out_of_sample = pd.DataFrame()
    time_df = pd.DataFrame()
    dataset_names = []
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
        print(dname + ' ' + str(real_dims) + 'd')
        dataset_names.append(dname + ' ' + str(real_dims) + 'd')
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
    # best_models_perf_in_sample.to_csv('in_sample.csv')
    # best_models_perf_out_of_sample.to_csv('out_sample.csv')
    plot_databframe_as_barplot(best_models_perf_in_sample, best_models_perf_out_of_sample, error_in_sample)
    # plot_databframe_as_barplot(best_models_perf_out_of_sample, None, False)


def plot_databframe_as_barplot(df_in, df_out, error_in):
    assert not any(df_in.index == df_out.index) is False
    leg_handles_dict = {
        'PROTEAN_{full}': ('tab:blue', '$PROTEAN_{full}$'),
        'PROTEAN_{fs}': ('tab:orange', '$PROTEAN_{fs}$'),
        'PROTEAN_{micencova}': ('tab:green', '$PROTEAN_{micencova}$'),
        'PROTEAN_{shap}': ('tab:red', '$PROTEAN_{shap}$'),
        'PROTEAN_{loda}': ('tab:purple', '$PROTEAN_{loda}$'),
    }
    leg_handles_arr = []
    colors = []
    for m in df_in.index:
        leg_handles_arr.append(leg_handles_dict[m][1])
        colors.append(leg_handles_dict[m][0])
    arr = np.arange(len(error_in.columns)) % 2
    errorbars = np.zeros((error_in.shape[0], 2, int(error_in.shape[1] / 2)), dtype=float)
    errorbars[:, 0, :] = error_in.iloc[:, arr == 0].values
    errorbars[:, 1, :] = error_in.iloc[:, arr == 1].values
    df_in.index = ['$' + x + '$' for x in df_in.index]
    df_out.index = ['$' + x + '$' for x in df_out.index]
    fig, axes = plt.subplots(figsize=(20,7), nrows=1, ncols=2)
    df_in.transpose().plot(ax=axes[0], kind='bar', zorder=3, rot=0, yerr=errorbars, capsize=5, grid=True, legend=None, color=colors)
    df_out.transpose().plot(ax=axes[1], kind='bar', zorder=3, rot=0, grid=True, legend=None)
    axes[0].legend(leg_handles_arr, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.3), fontsize=18)
    axes[0].set_yticks(np.arange(0, 1.1, .1))
    axes[1].set_yticks(np.arange(0, 1.1, .1))
    axes[0].tick_params(labelsize=14)
    axes[1].tick_params(labelsize=14)
    plt.subplots_adjust(hspace=0.65, wspace=0.65)
    plt.tight_layout()
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    fig_name = 'real_' if conf['type'] == 'real' else 'synthetic_'
    fig_name += conf['detector'] + '.png'
    plt.savefig(Path(output_folder, fig_name), dpi=300)
    plt.clf()


def plot_dataframe_as_table(best_model_perfs, in_sample):
    data_type = 'D_{IN}' if in_sample else 'D_{OUT}'
    y_detector_name = 'Y_{' + conf['detector'] + '}'
    title = 'AUC(' + y_detector_name.upper() + ', \hat{Y}_{M}, ' + data_type + ')'
    fig, ax = plt.subplots(nrows=1, ncols=1)
    table = ax.table(cellText=best_model_perfs.values, colLabels=best_model_perfs.columns,
                     colWidths=[0.4 for x in best_model_perfs.columns],
                     loc='top', rowLabels=best_model_perfs.index,
                     cellLoc='center', bbox=[0.15, 0.45, 0.8, 0.5])
    # table.scale(1.5, 1.5)
    table.set_fontsize(8)
    # table.auto_set_column_width(col=list(range(len(best_model_perfs.columns))))
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    ax.set_title('$' + title + '$')
    ax.axis('off')
    # plt.ylim((1, best_model_perfs.shape[1]))
    plt.show()


def get_best_models_perf_per_method(nav_file, in_sample):
    best_model_perfs = {}
    ci = {}
    error = {}
    protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    best_model_perfs['PROTEAN_{full}'], ci['PROTEAN_{full}'] = get_effectiveness_of_best_model(protean_ps_dict, False,
                                                                                               in_sample)
    best_model_perfs['PROTEAN_{fs}'], ci['PROTEAN_{fs}'] = get_effectiveness_of_best_model(protean_ps_dict, True,
                                                                                           in_sample)
    baselines_dir = nav_file[FileKeys.navigator_baselines_key]
    for method, data in baselines_dir.items():
        if method == 'random':
            continue
        method_name = 'PROTEAN_{' + method + '}'
        best_model_perfs[method_name], ci[method_name] = get_effectiveness_of_best_model(data, True, in_sample)
    for m in ci.keys():
        error[m] = [np.abs(ci[m][0] - best_model_perfs[m]), np.abs(ci[m][1] - best_model_perfs[m])]
        best_model_perfs[m] = [best_model_perfs[m]]
        ci[m] = [ci[m]]
    return pd.DataFrame(best_model_perfs).transpose(), pd.DataFrame(ci).transpose(), pd.DataFrame(error).transpose()


def get_time_per_method(nav_file):
    runtime = {}
    protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    protean_best_model, _ = PseudoSamplesMger(protean_ps_dict, 'roc_auc', True).get_best_model()
    runtime['protean'] = protean_best_model['feature_selection']['time']
    explanations = Path(nav_file[FileKeys.navigator_baselines_dir_key], FileNames.baselines_explanations_fname)
    with open(explanations) as json_file:
        explanations_dict = json.load(json_file)
        for method, data in explanations_dict.items():
            if method == 'random':
                continue
            method_name = 'PROTEAN_{' + method + '}'
            runtime[method_name] = round(data['time'], 2)
    return pd.DataFrame(runtime.values(), index=runtime.keys())


def get_effectiveness_of_best_model(ps_dict, fs, in_sample):
    effectiveness_key = 'effectiveness' if in_sample else 'hold_out_effectiveness'
    method_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs=fs)
    best_model, best_k = method_mger.get_best_model()
    conf_intervals = [round(x, 2) for x in best_model['confidence_intervals']]
    return round(best_model[effectiveness_key], 3), conf_intervals


def build_baseline_models():
    print(conf)
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    dataset_names = []
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] == 'synthetic')
        print('Evaluating explanations for Dataset', dname + ' ' + str(real_dims) + '-d')
        dataset_names.append(dname + ' ' + str(real_dims) + '-d')
        ConfigMger.setup_configs(nav_file[FileKeys.navigator_conf_path])
        ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], 'roc_auc', fs=True)
        baselines_dir = nav_file[FileKeys.navigator_baselines_dir_key]
        explanations = load_baseline_explanations(baselines_dir, MAX_FEATURES)
        run_baseline_explanations_in_automl(ps_mger, explanations, baselines_dir)


def run_baseline_explanations_in_automl(ps_mger, explanations, baselines_dir):
    sorted_k_confs = sorted(ps_mger.list_k_confs())
    reps_fold_inds = get_reps_folds_inds(ps_mger)

    for k in sorted_k_confs:
        print('Running with pseudo samples', k)
        train_dataset, test_dataset = get_datasets(ps_mger, k)
        for method, expl in explanations.items():
            print('----\nRunning method', method, 'with explanation', expl)
            method_output_dir = Path(baselines_dir, method, 'pseudo_samples' + str(k))
            method_output_dir.mkdir(parents=True, exist_ok=True)
            best_model = AutoML(method_output_dir).run_with_explanation(reps_fold_inds[k], train_dataset, expl)
            if holdout:
                best_model = test_best_model_in_hold_out(best_model, test_dataset)
            write_best_model(best_model, method_output_dir)


def write_best_model(best_model, output_dir):
    best_model_to_write = {'fs': {}}
    for m_id, m_data in best_model.items():
        best_model_to_write['fs'][m_id] = m_data.to_dict()
    with open(os.path.join(output_dir, FileNames.best_model_fname), 'w', encoding='utf-8') as f:
        f.write(json.dumps(best_model_to_write, indent=4, separators=(',', ': '), ensure_ascii=False))


def get_datasets(ps_mger, k):
    anomaly_col = DatasetConfig.get_anomaly_column_name()
    subspace_col = DatasetConfig.get_subspace_column_name()
    train_dataset_path = ps_mger.get_info_field_of_k(k, FileKeys.navigator_pseudo_samples_data_path)
    train_dataset = Dataset(train_dataset_path, anomaly_col, subspace_col)
    test_data = None
    if holdout:
        test_dataset_path = ps_mger.get_info_field_of_k(k, FileKeys.navigator_pseudo_samples_hold_out_data_key)
        test_data = Dataset(test_dataset_path, anomaly_col, subspace_col)
    return train_dataset, test_data


def test_best_model_in_hold_out(best_model, test_data):
    for m_id, conf in best_model.items():
        fsel = conf.get_fsel()
        clf = conf.get_clf()
        X_new = test_data.get_X().iloc[:, fsel.get_features()]
        predictions = clf.predict_proba(X_new)
        perf = metrics.calculate_metric(test_data.get_Y(), predictions, m_id)
        conf.set_hold_out_effectiveness(perf, m_id)
        best_model[m_id] = conf
    return best_model


def get_reps_folds_inds(ps_mger):
    sorted_k_confs = sorted(ps_mger.list_k_confs())
    reps_folds_inds_per_k = OrderedDict()
    for k in sorted_k_confs:
        ps_samples_dir = ps_mger.get_info_field_of_k(k, FileKeys.navigator_pseudo_sample_dir_key)
        repetitions = Path(ps_samples_dir, FileNames.indices_folder)
        rep_count = 0
        reps_folds_inds_per_k.setdefault(k, {})
        for f in os.listdir(repetitions):
            if not f.endswith('.json'):
                continue
            reps_folds_inds_per_k[k].setdefault(rep_count, {})
            with open(Path(repetitions, f)) as json_file:
                reps_folds_inds_per_k[k][rep_count] = json.load(json_file, object_pairs_hook=OrderedDict)
            rep_count += 1
    return reps_folds_inds_per_k


def test_baseline_explanations_noise():
    print(conf)
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    dataset_names = []
    anomaly_col = 'is_anomaly'
    subspace_col = None if conf['type'] == 'real' else 'subspaces'
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] == 'synthetic')
        print(dname + ' ' + str(real_dims) + '-d')
        dataset_names.append(dname + ' ' + str(real_dims) + '-d')
        ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], 'roc_auc', fs=True)
        train_data = Dataset(ps_mger.get_info_field_of_k(0, FileKeys.navigator_pseudo_samples_data_path), anomaly_col,
                             subspace_col)
        train_data_noise = add_noise_to_data(train_data)
        explanation_methods = ExplanationMethods(train_data)
        explanation_methods_noise = ExplanationMethods(train_data_noise)
        micencova_expl = get_topk_features_global_expl(explanation_methods.micencova_explanation, 10)
        micencova_expl_noise = get_topk_features_global_expl(explanation_methods_noise.micencova_explanation, 10)
        print('Noisy explanation', micencova_expl_noise)
        print(len(set(micencova_expl).intersection(micencova_expl_noise)) / len(
            set(micencova_expl).union(micencova_expl_noise)))


def add_noise_to_data(dataset):
    np.random.seed(0)
    noise_data = np.random.normal(0, 1, dataset.get_X().shape, )
    dataset_noise = pd.concat([dataset.get_df(), pd.DataFrame(noise_data)], axis=1)
    return Dataset(dataset_noise, dataset.get_anomaly_column_name(), dataset.get_subspace_column_name())


def get_topk_features_global_expl(baseline_func, max_features):
    global_expl = baseline_func()
    global_explanation_sorted = np.argsort(global_expl['global_explanation'])[::-1]
    return list(global_explanation_sorted[0:max_features])


if __name__ == '__main__':
    # test_baseline_explanations_noise()
    # exit()

    if build_models:
        build_baseline_models()
    else:
        compare_models()
