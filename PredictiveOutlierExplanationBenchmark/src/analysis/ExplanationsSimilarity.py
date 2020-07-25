from pathlib import Path
from matplotlib.font_manager import FontProperties
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)
from configpkg import ConfigMger, DatasetConfig
from holders.Dataset import Dataset
from utils.helper_functions import read_nav_files, sort_files_by_dim
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import FileKeys, FileNames
from analysis.comparison.comparison_utils import load_baseline_explanations, get_dataset_name
from collections import OrderedDict
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

pipeline = 'results_predictive'
MAX_FEATURES = 10

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'test'}
conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'test'}


# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'synthetic'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'synthetic'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'synthetic'}

# conf = {'path': Path('..', pipeline, 'lof'), 'detector': 'lof', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'iforest'), 'detector': 'iforest', 'type': 'real'}
# conf = {'path': Path('..', pipeline, 'loda'), 'detector': 'loda', 'type': 'real'}

def run_explanation_similarity_analysis():
    print(conf)
    dataset_names = []
    js_sims = pd.DataFrame()
    nav_files_json = sort_files_by_dim(read_nav_files(conf['path'], conf['type']))
    for dim, nav_file in nav_files_json.items():
        real_dims = dim - 1 - (conf['type'] == 'synthetic')
        dname = get_dataset_name(nav_file[FileKeys.navigator_original_dataset_path], conf['type'] != 'real')
        dataset_names.append(dname + ' ' + str(real_dims) + 'd')
        method_explanations = get_explanation_per_method(nav_file)
        js_sims = pd.concat([js_sims, jaccard_similarity(method_explanations)], axis=1)
    js_sims.columns = dataset_names
    sns.heatmap(js_sims, annot=True)
    plt.show()


def jaccard_similarity(method_explanations):
    jaccard_sim = {}
    for m_i, expl_i in method_explanations.items():
        for m_j, expl_j in method_explanations.items():
            js_sim = len(set(expl_i).intersection(expl_j)) / len(set(expl_i).union(expl_j))
            jaccard_sim[m_i + '_' + m_j] = round(js_sim,2)
    return pd.DataFrame(jaccard_sim.values(), index=jaccard_sim.keys())


def get_explanation_per_method(nav_file):
    methods_sel_features = {}
    protean_psmger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], 'roc_auc', fs=True)
    best_model, best_k = protean_psmger.get_best_model()
    methods_sel_features['$PROTEUS_{fs}$'] = best_model['feature_selection']['features']
    methods_explanations_file = Path(nav_file[FileKeys.navigator_baselines_dir_key],
                                     FileNames.baselines_explanations_fname)
    with open(methods_explanations_file) as json_file:
        explanations = json.load(json_file)
        for method, data in explanations.items():
            if method == 'random':
                continue
            if method == 'micencova':
                method = 'ca-lasso'
            features_sorted = np.argsort(np.array(data['global_explanation']))[::-1]
            method_name = '$PROTEUS_{' + method + '}$'
            methods_sel_features[method_name] = features_sorted[:MAX_FEATURES]
    return methods_sel_features


def analyze_train_test_splits():
    pass


def analyze_cv_splits():
    pass


if __name__ == '__main__':
    run_explanation_similarity_analysis()