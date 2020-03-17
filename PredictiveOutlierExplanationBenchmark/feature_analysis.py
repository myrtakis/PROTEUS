import pandas as pd
import matplotlib.pyplot as plt
from var_selection import run_var_selection
import numpy as np
import seaborn as sns
import os
from visualizers import *


# EXPERIMENTS #

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


def visualize_best_selected_features(results):
    pass


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


if __name__=='__main__':
    # run_feature_selection_algs()
    group1 = 'datasets/synthetic/hics/dimexp/g1dmin2dmax3'
    group2 = 'datasets/synthetic/hics/dimexp/g1dmin4dmax5'
    # visualize_ideal_features(group1, 'G1')
    visualize_ideal_features(group2, 'G2')