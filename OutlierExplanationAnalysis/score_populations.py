import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

rel_feature_subsets_col_name = 'subspaces'
results_dir = 'explanationExp'


def analyze_score_populations(path_dataset, expl_name, od_name, dataset_name):
    results_files = get_all_files(dataset_name, expl_name, od_name)
    ground_truth_df = pd.read_csv(path_dataset)
    rfs_ground_truth_dict = get_rel_feature_subsets(ground_truth_df)

    for f in results_files:
        results_df = pd.read_csv(f)


def get_rel_feature_subsets(df):
    rel_feature_subsets_str_dict = df.loc[df[rel_feature_subsets_col_name] != '-'][rel_feature_subsets_col_name].to_dict()
    rel_feature_subsets_list_dict = {}
    for k in rel_feature_subsets_str_dict.keys():
        rfs_str_list = rel_feature_subsets_str_dict[k].split(';')
        rfs_list = []
        for s in rfs_str_list:
            if '[' not in s or ']' not in s:
                continue
            fs_str = s[s.rfind('[')+1: s.rfind(']')]
            rfs_list.append(list(map(int, fs_str.split())))
        rel_feature_subsets_list_dict[k] = rfs_list
    return rel_feature_subsets_list_dict


def get_all_files(dataset_name, expl_name, od_name):
    pref_files = []
    for path, subdirs, files in os.walk(results_dir):
        for name in files:
            if expl_name not in name or od_name not in name or dataset_name not in path or not name.endswith('.csv'):
                continue
            pref_files.append(os.path.join(path, name))
    return pref_files


if __name__ == "__main__":
    analyze_score_populations('datasets/breast_lof_031_010.csv', 'beam', 'lof', 'breast')
