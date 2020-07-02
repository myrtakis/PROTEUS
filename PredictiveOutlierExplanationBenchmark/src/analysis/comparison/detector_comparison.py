from matplotlib.font_manager import FontProperties
from configpkg import ConfigMger, DatasetConfig
from holders.Dataset import Dataset
from models.OutlierDetector import Detector
from utils.helper_functions import read_nav_files, sort_files_by_dim
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys
from analysis.comparison.comparison_utils import load_baseline_explanations, get_dataset_name
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_FEATURES = 10

# conf = {'path': '../results_normal/lof', 'detector': 'lof', 'holdout': False, 'type': 'synthetic'}
# conf = {'path': '../results_normal/iforest', 'detector': 'iforest', 'holdout': False, 'type': 'synthetic'}
conf = {'path': '../results_normal/lof', 'detector': 'lof', 'holdout': False, 'type': 'real'}


def compare_methods():
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    explanation_perfs = []
    methods = []
    dataset_names = []
    for dim, nav_file in nav_files_json.items():
        real_dims = dim-1-(conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] == 'synthetic')
        dataset_names.append(dname + ' ' + str(real_dims) + '-d')
        ConfigMger.setup_configs(nav_file[FileKeys.navigator_conf_path])
        ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], 'roc_auc', fs=True)
        detector = get_detector_model()
        dataset = get_dataset(ps_mger)
        explanations = {}
        explanations.update(get_protean_features(ps_mger))
        explanations.update(load_baseline_explanations(nav_file[FileKeys.navigator_baselines_dir_key], MAX_FEATURES))
        if len(methods) == 0:
            methods = sorted(explanations.keys())
        explanation_perfs.append(auc_perfs_using_explanation(detector, dataset, explanations))
    return pd.DataFrame(np.array(explanation_perfs).T, index=methods, columns=dataset_names)


def auc_perfs_using_explanation(detector, dataset, explanations):
    perfs = []
    for method in sorted(explanations.keys()):
        detector.train(dataset.get_X().iloc[:, explanations[method]])
        perfs.append(roc_auc_score(dataset.get_Y(), detector.get_scores_in_train()))
    return perfs


def get_protean_features(ps_mger):
    best_model, k = ps_mger.get_best_model()
    return {'protean': best_model['feature_selection']['features']}


def get_detector_model():
    for d in Detector.init_detectors():
        if d.get_id() == conf['detector']:
            return d
    assert False


def get_dataset(ps_mger):
    dataset_path_det = ps_mger.get_info_field_of_k(0)
    return Dataset(dataset_path_det, DatasetConfig.get_anomaly_column_name(), DatasetConfig.get_subspace_column_name())


def visualize_results_as_table(perfomance_df):
    perfomance_df = perfomance_df.round(3)
    hold_out = 'yes' if conf['holdout'] else 'no'
    title = 'Explaining ' + conf['detector'].upper() + ' (hold out: ' + hold_out + ')'
    fig, ax = plt.subplots(nrows=1, ncols=1)

    table = ax.table(cellText=perfomance_df.values, colLabels=perfomance_df.columns,
                     loc='top', rowLabels=perfomance_df.index,
                     cellLoc='center', bbox=[0.15, 0.45, 0.8, 0.5])

    table.scale(0.8, 2.5)

    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    ax.set_title(title)
    ax.axis('off')
    plt.ylim((1, perfomance_df.shape[1]))
    plt.show()


if __name__ == '__main__':
    performances_df = compare_methods()
    visualize_results_as_table(performances_df)
    print()
