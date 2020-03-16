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


def visualize_ideal_features(features, min_dataset_dim):
    pass


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
    run_feature_selection_algs()