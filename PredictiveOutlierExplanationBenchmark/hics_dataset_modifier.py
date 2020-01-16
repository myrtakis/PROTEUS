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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ddir', '--datasets_dir', required=True)
    parser.add_argument('-mfile', '--main_file')
    parser.add_argument('-g', '--groups', help='How many subspace groups per dimensionality', required=True, type=int)
    parser.add_argument('-maxdim', '--maximum_dimensionality', required=True, type=int)
    parser.add_argument('-mindim', '--minimum_dimensionality', required=True, type=int)
    parser.add_argument('-s', '--save_dir', required=True)
    args = parser.parse_args()
    modify_datasets(args)
