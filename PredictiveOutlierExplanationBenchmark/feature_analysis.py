import json
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
from var_selection import run_var_selection
import numpy as np
import seaborn as sns
import os
from visualizers import *


# EXPERIMENTS FOR FEATURE SELECTION ALGORITHMS #

def run_feature_selection_algs():
    dir = 'datasets/synthetic/hics/dimexp/g1dmin2dmax3/ocluster5'
    # dir = 'datasets/real/lof_based'
    for f in os.listdir(dir):
        print(f)
        # if f != 'hics_20_g1_dmin2_dmax3.csv':
        #     continue
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(dir, f))
        # subspaces = df.loc[df['subspaces'] != '-', 'subspaces']
        # print(f, set(list(subspaces.to_dict().values())))
        # df = df.drop(columns=['subspaces'])
        # finalDf = run_ses(df.drop(columns='is_anomaly'), df['is_anomaly'])
        finalDf = df.iloc[:, [0,1,2, df.shape[1]-2]]
        visualize_selected_features(finalDf, list(range(3)), '')
        # manifold_visualizations(finalDf, '')
        # pca_visualization(finalDf, '')
        # ica_visualization(finalDf, '')
        # tsne_visualization(finalDf, '')
        # visualize_2d(finalDf, '')
        # umap_visualization(finalDf, '')
        # break


def run_ses(X_train, Y_train):
    params = {'alpha': [0.2],
              'max_k': [5]}
    features = None
    for a in params['alpha']:
        for k in params['max_k']:
            features = run_var_selection('ses', {'alpha': a, 'max_k': k}, X_train, Y_train)
    print(X_train.columns)
    finalDf = pd.concat([X_train.iloc[:, features], Y_train], axis=1)
    return finalDf


def run_lasso(df):
    pass


# ======================================================#


# IDEAL FEATURE VISUALIZATION #


def visualize_ideal_features(group_dir, group_id):
    files_path = {}
    for r, d, f in os.walk(group_dir):
        for file in f:
            if file.endswith('.csv'):
                files_path[os.path.basename(r)] = os.path.join(r, file)
    for cluster_size, dataset_path in files_path.items():
        df = pd.read_csv(dataset_path)
        subspaces_as_int_list = get_subspaces_as_list(df)
        df = df.drop(columns=['subspaces'])
        fig_path = os.path.join('visualizations', 'synthetic', group_id, 'ideal', cluster_size)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        for subspace in subspaces_as_int_list:
            print('Visualizing ideal path of cluster', cluster_size, 'and subspace', subspace)
            cols = ['is_anomaly']
            cols.extend(df.columns[subspace])
            subDf = df.loc[:, cols]
            visualize_selected_features(subDf, subspace, fig_path)


# ======================================================#


# VISUALIZE FEATURES SELECTED BY THE BEST CONFIGURATION #


def find_best_selected_features(results_file, dataset_path, savedir, fsel_alg=None):
    best_config = get_best_performance(results_file, fsel_alg)
    df = pd.read_csv(dataset_path)
    sel_features = list(map(int, best_config['sel_features'].replace('[', '').replace(']', '').split()))
    subDf = pd.concat([df.iloc[:, sel_features], df['is_anomaly']], axis=1)
    visualize_selected_features(subDf, sel_features, savedir)


def get_best_performance(results_file, fsel_alg=None):
    metric = 'roc_auc'
    best_performances = {}
    with open(results_file) as json_file:
        results = json.load(json_file)
        for rep, data_in_rep in results.items():
            for conf, conf_data in data_in_rep[metric].items():
                if 'none' in conf or (fsel_alg is not None and fsel_alg not in conf):
                    continue
                best_performances.setdefault(conf_data['performance'], [])
                best_performances[conf_data['performance']].append(conf_data)
    best_performances = OrderedDict(sorted(best_performances.items(), reverse=True))
    best_config = None
    min_features = None
    for config, config_data in best_performances.items():
        for c in config_data:
            feature_num = len(c['sel_features'].replace('[','').replace(']', '').split())
            if min_features is None or feature_num < min_features:
                best_config = c
                min_features = len(c['sel_features'])
        # Break since the best configuration is ranked first
        break
    return best_config


# ======================================================#


# UTIL FUNCTIONS #


def get_subspaces_as_list(df):
    subspaces = set(df.loc[df['subspaces'] != '-', 'subspaces'].values)
    subspaces_as_list_ind = []
    all_rel_features = set()
    for s in subspaces:
        subspace = list(map(int, s[s.index('[')+1: s.index(']')].split()))
        subspaces_as_list_ind.append(subspace)
        all_rel_features = all_rel_features.union(subspace)
    subspaces_as_list_ind.append(list(all_rel_features))
    return subspaces_as_list_ind


def check_if_normalized(f):
    df = pd.read_csv(f)
    for col in df.columns:
        print(col, 'range', min(df[col].values), max(df[col].values))


def get_df_by_log(log_file_path):
    with open(log_file_path) as json_file:
        log = json.load(json_file)
        print(log_file_path)
        with open(log['config']) as conf_json_file:
            conf = json.load(conf_json_file)
            return pd.read_csv(conf['datasets']['d1']['dataset_path'])



# ======================================================#


# MAIN #


if __name__=='__main__':
    # run_feature_selection_algs()

    # Visualize Ideal Features

    # group1 = 'datasets/synthetic/hics/dimexp/g1dmin2dmax3'
    # group2 = 'datasets/synthetic/hics/dimexp/g1dmin4dmax5'
    # visualize_ideal_features(group1, 'G1')
    # visualize_ideal_features(group2, 'G2')

    # Visualize specific features

    # features = [' var_0003',' var_0036',' var_0023',' var_0069',' var_0025',' var_0075',' var_0024',' var_0089',' var_0041',' var_0006',' var_0060',' var_0031',' var_0019',' var_0059',' var_0028',' var_0035',' var_0030',' var_0004',' var_0040',' var_0048',' var_0037',' var_0047',' var_0002',' var_0014',' var_0029','is_anomaly']
    # df = pd.read_csv('datasets/synthetic/hics/dimexp/g1dmin2dmax3/ocluster50/hics_100_g1_dmin2_dmax3.csv').loc[:, features]
    # tsne_visualization(df, [], 'visualizations/jadFeatures/g1/ocluster50/tsne')
    # pca_visualization(df, [], 'visualizations/jadFeatures/g1/ocluster50/pca')

    # Visualize Features of best configurations

    # find_best_selected_features('results/synthetic/dimexp/g1dmin2dmax3/ocluster5/hics100_g1_dmin2_dmax3.json',
    #                             'datasets/synthetic/hics/dimexp/g1dmin2dmax3/ocluster5/hics_100_g1_dmin2_dmax3.csv',
    #                             'visualizations/best_configs/synthetic/g1/original_data')
    # find_best_selected_features('results/synthetic/dimexp/g1dmin2dmax3/ocluster25/hics100_g1_dmin2_dmax3.json',
    #                             'datasets/synthetic/hics/dimexp/g1dmin2dmax3/ocluster25/hics_100_g1_dmin2_dmax3.csv',
    #                             'visualizations/best_configs/synthetic/g1/ocluster25')
    # find_best_selected_features('results/synthetic/dimexp/g1dmin2dmax3/ocluster50/hics100_g1_dmin2_dmax3.json',
    #                             'datasets/synthetic/hics/dimexp/g1dmin2dmax3/ocluster50/hics_100_g1_dmin2_dmax3.csv',
    #                             'visualizations/best_configs/synthetic/g1/ocluster50')

    # find_best_selected_features('results/real/lof_based/breast_lof_results.json',
    #                             'datasets/real/lof_based/breast_lof.csv',
    #                             'visualizations/best_configs/real/lof_based/breast')
    # find_best_selected_features('results/real/lof_based/breast_diag_lof_results.json',
    #                             'datasets/real/lof_based/breast_diagnostic_lof.csv',
    #                             'visualizations/best_configs/real/lof_based/breast_diag')
    find_best_selected_features('results/real/lof_based/electricity_lof_results.json',
                                'datasets/real/lof_based/electricity_lof.csv',
                                'visualizations/best_configs/real/lof_based/electricity')

