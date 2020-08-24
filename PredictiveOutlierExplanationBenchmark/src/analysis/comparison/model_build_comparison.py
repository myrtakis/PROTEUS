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
from pipeline.automl.automl_constants import MAX_FEATURES


pipeline = 'results_predictive'
holdout = True
build_models = False  # compare the built models

test_confs = [
        #{'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'},
        {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}
    ]


synth_confs =[
    {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'synthetic'},
    {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}
]

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}

confs_to_analyze = synth_confs


def compare_models():
    pred_perfs_dict = OrderedDict()
    for conf in confs_to_analyze:
        print(conf)
        pred_perfs_dict[conf['detector']] = best_models(conf)
    plot_results(pred_perfs_dict)


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


def plot_results(pred_perfs_dict):
    fig, axes = plt.subplots(figsize=(11, 4.5), nrows=2, ncols=3)
    for j, (det, pred_perf) in enumerate(pred_perfs_dict.items()):
        df_in = pred_perf['best_models_perf_in_sample']
        df_out = pred_perf['best_models_perf_out_of_sample']
        error_in = pred_perf['error_in_sample']
        assert not any(df_in.index == df_out.index) is False
        add_dataframe_to_axes(df_in, axes[0, j], error_in)
        add_dataframe_to_axes(df_out, axes[1, j])
        axes[0, j].set_xticklabels([])
        if det == 'iforest':
            det = 'iForest'
        else:
            det = det.upper()
        axes[0, j].set_title(det, fontsize=15, fontweight='bold')
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=13)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0, top=0.78)
    output_folder = Path('..', 'figures', 'results')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_folder, 'synthetic_auc.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()


def add_dataframe_to_axes(df, ax, error_in=None):
    leg_handles_dict = {
        'PROTEUS_{full}': ('tab:blue', '$PROTEUS_${full}$'),
        'PROTEUS_{fs}': ('tab:orange', '$PROTEUS_${fs}$'),
        'PROTEUS_{ca-lasso}': ('tab:green', '$PROTEUS_${ca-lasso}$'),
        'PROTEUS_{shap}': ('tab:red', 'PROTEUS$_{shap}$'),
        'PROTEUS_{loda}': ('tab:purple', 'PROTEUS$_{loda}$'),
        'PROTEUS_{random}': ('cyan', 'PROTEUS$_{random}$')
    }
    leg_handles_arr = []
    colors = []
    for m in df.index:
        leg_handles_arr.append(leg_handles_dict[m][1])
        colors.append(leg_handles_dict[m][0])
    errorbars = None
    if error_in is not None:
        arr = np.arange(len(error_in.columns)) % 2
        errorbars = np.zeros((error_in.shape[0], 2, int(error_in.shape[1] / 2)), dtype=float)
        errorbars[:, 0, :] = error_in.iloc[:, arr == 0].values
        errorbars[:, 1, :] = error_in.iloc[:, arr == 1].values
    df.index = [x.replace('_', '$_') + '$' for x in df.index]
    df.transpose().plot(ax=ax, kind='bar', zorder=3, rot=25, yerr=errorbars, capsize=2, width=0.7,
                        linewidth=0.08, ecolor='darkslategrey', grid=True, legend=None, color=colors)
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.set_ylabel('Mean AUC')
    #ax[1].tick_params(labelsize=18)


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
    best_model_perfs['PROTEUS_{full}'], ci['PROTEUS_{full}'] = get_effectiveness_of_best_model(protean_ps_dict, False,
                                                                                               in_sample)
    best_model_perfs['PROTEUS_{fs}'], ci['PROTEUS_{fs}'] = get_effectiveness_of_best_model(protean_ps_dict, True,
                                                                                           in_sample)
    baselines_dir = nav_file[FileKeys.navigator_baselines_key]
    for method, data in baselines_dir.items():
        if method == 'random':
            continue
        if method == 'micencova':
            method = 'ca-lasso'
        method_name = 'PROTEUS_{' + method + '}'
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
            method_name = 'PROTEUS_{' + method + '}'
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
