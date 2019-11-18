import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from detectors import lof

rel_feature_subsets_col_name = 'subspaces'
estimated_feature_subsets_col_name = '__REL_SUBSPACES'
outlier_col_name = 'is_anomaly'
results_dir = 'explanationExp'
dataset = 'datasets/breast_lof_031_010.csv'


def analyze_results_populations(path_dataset, expl_name, od_name, dataset_name):
    results_files = get_all_files(dataset_name, expl_name, od_name)
    ground_truth_df = pd.read_csv(path_dataset)
    inlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 0].index)
    outlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 1].index)
    rfs_results_dict = dict.fromkeys(ground_truth_df.index, [])
    for f in results_files:
        results_df = pd.read_csv(f)
        rfs = get_rel_feature_subsets(results_df, estimated_feature_subsets_col_name)
        for point_id in rfs.keys():
            rfs_results_dict[point_id].append(rfs[point_id])
            if point_id == 1:
                print(rfs[point_id])
                #print(rfs_results_dict[point_id])


def analyze_ground_truth_population(path_dataset):
    ground_truth_df = pd.read_csv(path_dataset)
    rfs_ground_truth_dict = get_rel_feature_subsets(ground_truth_df, rel_feature_subsets_col_name)
    mean_population = score_population_mean(ground_truth_df, rfs_ground_truth_dict)
    inlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 0].index)
    outlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 1].index)
    #plot_score_population(mean_population[outlier_inds], mean_population[inlier_inds])
    return mean_population


def score_population_mean(df, rfs_dict):
    mean_scores = np.array([0.0] * len(df))
    inlier_inds = list(df[df[outlier_col_name] == 0].index)
    rfs_no = 0
    for point_id in rfs_dict.keys():
        for rfs in rfs_dict[point_id]:
            rfs_no += 1
            sub_df = df.iloc[:, rfs]
            scores = lof(sub_df, {'knn': 15})
            mean_scores[point_id] += scores[point_id]
            mean_scores[inlier_inds] += scores[inlier_inds]
        mean_scores[point_id] = mean_scores[point_id] / len(rfs_dict[point_id])
    mean_scores[inlier_inds] = mean_scores[inlier_inds] / rfs_no
    return mean_scores


def plot_score_population(outlier_population, inlier_population):
    outlier_scores_sorted = sorted(outlier_population, reverse=True)
    inlier_scores_sorted = sorted(inlier_population, reverse=True)

    print(outlier_scores_sorted)
    print(inlier_scores_sorted)

    #plt.plot(range(len(outlier_population)), outlier_scores_sorted, '-*')
    #plt.plot(range(len(inlier_population)), inlier_scores_sorted, '-^')
    plt.hist(outlier_scores_sorted, label='outlier scores')
    plt.hist(inlier_scores_sorted, label='inlier scores')
    plt.xlabel('Scores')
    plt.legend(loc='best')
    #plt.show()


def summarization_score_population_mean(df, rfs_ground_truth_dict):
    return None


def get_rel_feature_subsets(df, feature_subsets_column):
    rel_feature_subsets_str_dict = df.loc[df[feature_subsets_column] != '-'][feature_subsets_column].to_dict()
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
    #analyze_ground_truth_population(dataset)
    analyze_results_populations(dataset, 'beam', 'lof', 'breast')
