from itertools import chain

import pandas as pd
import argparse
import os
import numpy as np
import collections
from utils import subspace_to_list
from utils import sort_datasets_dim

subspace_column = 'subspaces'
is_anomaly_column = 'is_anomaly'


def modify_datasets(args):
    outliers_to_keep = None
    datasets_dim_sorted = sort_datasets_dim(get_files(args.datasets_dir))
    for f in datasets_dim_sorted:
        if f == args.main_file:
            continue
        df = pd.read_csv(f)
        outlier_indexes = np.where(df[is_anomaly_column] == 1)[0]
        subspace_map = get_subspaces_and_indexes(df, outlier_indexes, args.minimum_dimensionality, args.maximum_dimensionality)
        if outliers_to_keep is None:
            outliers_to_keep = find_outliers_to_keep(args.groups, subspace_map)
        outliers_to_remove = list(set(outlier_indexes) - set(outliers_to_keep))
        final_df = df.drop(outliers_to_remove)
        dataset_name = os.path.splitext(os.path.basename(f))[0] + '_g' + str(args.groups) \
                       + '_dmin' + str(args.minimum_dimensionality) + '_dmax' + str(args.maximum_dimensionality) + '.csv'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        output_file = os.path.join(args.save_dir, dataset_name)
        final_df.to_csv(output_file, index=False)


def find_outliers_to_keep(groups, subspace_map):
    outlier_to_keep = set()
    for dim, subspaces in subspace_map.items():
        subspace_keys = list(subspaces.keys())
        for i in range(groups):
            outlier_to_keep = outlier_to_keep.union(subspaces[subspace_keys[i]])
    return outlier_to_keep


def find_subspaces_to_keep(groups, subspace_map):
    subspaces_to_keep = []
    for dim, subspaces in subspace_map.items():
        counter = 0
        for sid_pids in subspaces.items():
            if counter == groups:
                break
            subspaces_to_keep.append(sid_pids)
            counter += 1
    return subspaces_to_keep


def get_subspaces_and_indexes(df, outlier_indexes, min_dim, max_dim):
    outlier_df = df.iloc[outlier_indexes, :]
    subspace_map = {}
    counter = 0
    for subspacestr in outlier_df[subspace_column]:
        point_index = outlier_indexes[counter]
        counter += 1
        valid_subspaces = subspace_to_list(subspacestr, min_dim, max_dim)
        for sid, dim in valid_subspaces.items():
            if dim not in subspace_map:
                subspace_map[dim] = {}
            if sid not in subspace_map[dim]:
                subspace_map[dim][sid] = []
            subspace_map[dim][sid].append(point_index)
    return subspace_map


def get_files(dir_path):
    fileslist = []
    if not os.path.isdir(dir_path):
        return [dir_path]
    allfiles = os.listdir(dir_path)
    for f in allfiles:
        if f.endswith('.csv'):
            fileslist.append(os.path.join(dir_path, f))
    return fileslist


def create_dim_exp_from_dataset(args):
    df = pd.read_csv(args.main_file)
    numeric_features = df.shape[1] - 2
    outlier_indexes = np.where(df[is_anomaly_column] == 1)[0]
    sub_map = get_subspaces_and_indexes(df, outlier_indexes, args.minimum_dimensionality, args.maximum_dimensionality)
    subspaces_to_keep = find_subspaces_to_keep(args.groups, sub_map)
    outliers_to_keep = set(chain.from_iterable([x[1] for x in subspaces_to_keep]))
    outliers_to_remove = set(outlier_indexes) - outliers_to_keep
    base_df = get_base_df_target_subspaces(df, subspaces_to_keep, outliers_to_remove)
    target, subspaces = base_df[is_anomaly_column], base_df[subspace_column]
    base_df = base_df.drop(columns=[is_anomaly_column, subspace_column])
    df = df.drop(index=outliers_to_remove)
    features_per_dataset = int(np.ceil(numeric_features / args.sub_datasets_num))
    features_pool = get_features_pool_shuffled(subspaces_to_keep, numeric_features)
    col_names = list(base_df.columns)
    for i in range(args.sub_datasets_num):
        features_to_add = (i+1) * features_per_dataset - base_df.shape[1]
        if base_df.shape[1] + features_to_add > numeric_features:
            features_to_add = numeric_features - base_df.shape[1]
        col_names.extend(list(df.iloc[:, features_pool[0:features_to_add]].columns))
        base_df = pd.concat([base_df, df.iloc[:, features_pool[0:features_to_add]]], ignore_index=True, axis=1)
        base_df.columns = col_names
        features_pool = np.delete(features_pool, list(range(features_to_add)))
        dataset_name = 'hics_' + str(base_df.shape[1]) + '_g' + str(args.groups) \
                       + '_dmin' + str(args.minimum_dimensionality) + '_dmax' + str(
            args.maximum_dimensionality) + '.csv'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        output_file = os.path.join(args.save_dir, dataset_name)
        tmpdf = base_df.copy()
        tmpdf[is_anomaly_column] = target
        tmpdf[subspace_column] = subspaces
        tmpdf.to_csv(output_file, index=False)


def get_base_df_target_subspaces(df, subspaces_to_keep, outliers_to_remove):
    base_df = pd.DataFrame()
    col_names = []
    target = df[is_anomaly_column]
    subspaces = df[subspace_column].copy()
    for item in subspaces_to_keep:
        subspace_feautures = list(map(int, item[0].split()))
        feature_sig = np.arange(base_df.shape[1], base_df.shape[1] + len(subspace_feautures))
        subspaces.iloc[item[1]] = str(feature_sig) + ';'
        tmpdf = df.iloc[:, subspace_feautures]
        col_names.extend(list(tmpdf.columns))
        base_df = pd.concat([base_df, tmpdf], ignore_index=True, axis=1)
    base_df = pd.concat([base_df, target, subspaces], ignore_index=True, axis=1)
    col_names.append(is_anomaly_column)
    col_names.append(subspace_column)
    base_df.columns = col_names
    return base_df.drop(index=outliers_to_remove)


def get_features_pool_shuffled(subspaces_to_keep, num_of_features):
    feature_pool = set(np.arange(0, num_of_features))
    for item in subspaces_to_keep:
        for f in list(map(int, item[0].split())):
            feature_pool.remove(f)
    feature_pool = np.array(list(feature_pool))
    np.random.seed(0)
    np.random.shuffle(feature_pool)
    return feature_pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ddir', '--datasets_dir', default=None)
    parser.add_argument('-mfile', '--main_file')
    parser.add_argument('-g', '--groups', help='How many subspace groups per dimensionality', required=True, type=int)
    parser.add_argument('-maxdim', '--maximum_dimensionality', required=True, type=int)
    parser.add_argument('-mindim', '--minimum_dimensionality', required=True, type=int)
    parser.add_argument('-subdata', '--sub_datasets_num', type=int)
    parser.add_argument('-s', '--save_dir', required=True)
    args = parser.parse_args()
    if args.datasets_dir is not None:
        modify_datasets(args)
    else:
        create_dim_exp_from_dataset(args)