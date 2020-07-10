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
conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}


def compare_models():
    print(conf)
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    best_models_perf_in_sample = pd.DataFrame()
    best_models_perf_out_of_sample = pd.DataFrame()
    dataset_names = []
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] == 'synthetic')
        print(dname + ' ' + str(real_dims) + '-d')
        dataset_names.append(dname + ' ' + str(real_dims) + '-d')
        best_models_perf_in_sample = pd.concat(
            [best_models_perf_in_sample, get_best_models_perf_per_method(nav_file, True)], axis=1
        )
        best_models_perf_out_of_sample = pd.concat(
            [best_models_perf_out_of_sample, get_best_models_perf_per_method(nav_file, False)], axis=1
        )
    best_models_perf_in_sample.columns = dataset_names
    best_models_perf_out_of_sample.columns = dataset_names
    plot_dataframe(best_models_perf_in_sample, True)
    plot_dataframe(best_models_perf_out_of_sample, False)


def plot_dataframe(best_model_perfs, in_sample):
    if in_sample:
        title = 'In Sample AUC'
    else:
        title = 'Out of Sample AUC'
    title += ' (' + conf['detector'] + ')'
    fig, ax = plt.subplots(nrows=1, ncols=1)
    table = ax.table(cellText=best_model_perfs.values, colLabels=best_model_perfs.columns,
                     loc='top', rowLabels=best_model_perfs.index,
                     cellLoc='center', bbox=[0.15, 0.45, 0.8, 0.5])
    table.scale(0.8, 2.5)
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    ax.set_title(title)
    ax.axis('off')
    plt.ylim((1, best_model_perfs.shape[1]))
    plt.show()


def get_best_models_perf_per_method(nav_file, in_sample):
    best_model_perfs = {}
    protean_ps_dict = nav_file[FileKeys.navigator_pseudo_samples_key]
    best_model_perfs['full'] = get_effectiveness_of_best_model(protean_ps_dict, False, in_sample)
    best_model_perfs['protean'] = get_effectiveness_of_best_model(protean_ps_dict, True, in_sample)
    baselines_dir = nav_file[FileKeys.navigator_baselines_key]
    for method, data in baselines_dir.items():
        best_model_perfs[method] = get_effectiveness_of_best_model(data, True, in_sample)
    return pd.DataFrame(best_model_perfs.values(), index=best_model_perfs.keys())


def get_effectiveness_of_best_model(ps_dict, fs, in_sample):
    effectiveness_key = 'effectiveness' if in_sample else 'hold_out_effectiveness'
    method_mger = PseudoSamplesMger(ps_dict, 'roc_auc', fs=fs)
    best_model, best_k = method_mger.get_best_model()
    conf_intervals = [round(x,2)for x in best_model['confidence_intervals']]
    output_subpart1 = str(round(best_model[effectiveness_key], 3))
    output_subpart2 = '' if not in_sample else ' ' + str(conf_intervals)
    output = output_subpart1 + output_subpart2
    return output


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
        train_data = Dataset(ps_mger.get_info_field_of_k(0, FileKeys.navigator_pseudo_samples_data_path), anomaly_col, subspace_col)
        train_data_noise = add_noise_to_data(train_data)
        explanation_methods = ExplanationMethods(train_data)
        explanation_methods_noise = ExplanationMethods(train_data_noise)
        micencova_expl = get_topk_features_global_expl(explanation_methods.micencova_explanation, 10)
        micencova_expl_noise = get_topk_features_global_expl(explanation_methods_noise.micencova_explanation, 10)
        print('Noisy explanation', micencova_expl_noise)
        print(len(set(micencova_expl).intersection(micencova_expl_noise)) / len(set(micencova_expl).union(micencova_expl_noise)))


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

