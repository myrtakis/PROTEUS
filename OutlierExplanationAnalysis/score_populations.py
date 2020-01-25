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
xlim = 0


def analyze_results_populations_points(path_dataset, expl_name, od_name, dataset_name):
    results_files = get_all_files(dataset_name, expl_name, od_name)
    ground_truth_df = pd.read_csv(path_dataset)
    outlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 1].index)
    fs_results_dict = {}
    for f in results_files:
        results_df = pd.read_csv(f)
        for point_id in outlier_inds:
            best_fs = get_point_best_estimated_feature_subset(results_df, point_id, estimated_feature_subsets_col_name)
            if point_id not in fs_results_dict:
                fs_results_dict[point_id] = [best_fs]
            else:
                fs_results_dict[point_id].append(best_fs)
    print('beam', fs_results_dict)
    mean_population = score_population_mean(ground_truth_df, fs_results_dict)
    global xlim
    xlim = max(xlim, max(mean_population))
    return mean_population


def analyze_results_populations_summarization(path_dataset, expl_name, od_name, dataset_name):
    results_files = get_all_files(dataset_name, expl_name, od_name)
    ground_truth_df = pd.read_csv(path_dataset)
    outlier_inds = list(ground_truth_df[ground_truth_df[outlier_col_name] == 1].index)
    fs_results_dict = {}
    for f in results_files:
        results_df = pd.read_csv(f)
        fs_total_results = get_rel_feature_subsets(results_df, estimated_feature_subsets_col_name)
        print(fs_total_results)
        for point_id in outlier_inds:
            best_fs = max_point_fs(point_id, fs_total_results[point_id], ground_truth_df)
            if point_id not in fs_results_dict:
                fs_results_dict[point_id] = [best_fs]
            else:
                fs_results_dict[point_id].append(best_fs)
    mean_population = score_population_mean(ground_truth_df, fs_results_dict)
    global xlim
    xlim = max(xlim, max(mean_population))
    return mean_population


def analyze_ground_truth_population(path_dataset):
    ground_truth_df = pd.read_csv(path_dataset)
    rfs_ground_truth_dict = get_rel_feature_subsets(ground_truth_df, rel_feature_subsets_col_name)
    mean_population = score_population_mean(ground_truth_df, rfs_ground_truth_dict)
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
    global xlim
    xlim = max(xlim, max(mean_scores))
    return mean_scores


def plot_score_population(outlier_population, inlier_population, title):
    outlier_scores_sorted = sorted(outlier_population, reverse=True)
    inlier_scores_sorted = sorted(inlier_population, reverse=True)

    #plt.plot(range(len(outlier_population)), outlier_scores_sorted, '-*')
    #plt.plot(range(len(inlier_population)), inlier_scores_sorted, '-^')
    plt.hist(outlier_scores_sorted, label='outlier scores')
    plt.hist(inlier_scores_sorted, label='inlier scores')
    global xlim
    plt.xlim(0, xlim)
    plt.xticks(np.arange(0, xlim, 0.5))
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('figures/' + title + '.png', dpi=300)
    plt.clf()
    #plt.show()


def max_point_fs(point_id, fs_list, df):
    max_fs = None
    max_score = None
    for fs in fs_list:
        sub_df = df.iloc[:, fs]
        score = lof(sub_df, {'knn': 15})[point_id]
        if max_score is None or max_score < score:
            max_score = score
            max_fs = fs
    return max_fs



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


def get_point_best_estimated_feature_subset(df, point_id, feature_subsets_column):
    rel_feature_subsets_str = df.loc[point_id][feature_subsets_column]
    parts = rel_feature_subsets_str.split(';')
    max_score = None
    max_fs = None
    for p in parts:
        if p is None or len(p) == 0:
            continue
        fs_str = p[p.rfind('[') + 1: p.rfind(']')].strip()
        score = float(p[p.rfind(']') + 1: len(p)].strip())
        fs_list = list(map(int, fs_str.split()))
        if max_score is None:
            max_score = score
            max_fs = fs_list
        elif max_score < score:
            max_score = score
            max_fs = fs_list
    return max_fs


def get_all_files(dataset_name, expl_name, od_name):
    pref_files = []
    for path, subdirs, files in os.walk(results_dir):
        for name in files:
            if expl_name not in name or od_name not in name or dataset_name not in path or not name.endswith('.csv'):
                continue
            pref_files.append(os.path.join(path, name))
    return pref_files


def get_outlier_inlier_inds(dataset_path):
    df = pd.read_csv(dataset_path)
    outlier_inds = list(df[df[outlier_col_name] == 1].index)
    inlier_inds = list(df[df[outlier_col_name] == 0].index)
    return outlier_inds, inlier_inds


if __name__ == "__main__":
    outlier_inds, inlier_inds = get_outlier_inlier_inds(dataset)
    gt_mean_population = analyze_ground_truth_population(dataset)
    beam_mean_population = analyze_results_populations_points(dataset, 'beam', 'lof', 'breast')
    lookout_mean_population = analyze_results_populations_summarization(dataset, 'lookout', 'lof', 'breast')
    print(beam_mean_population)
    print(lookout_mean_population)
    plot_score_population(gt_mean_population[outlier_inds], gt_mean_population[inlier_inds], 'Ground_Truth_Breast')
    plot_score_population(beam_mean_population[outlier_inds], beam_mean_population[inlier_inds], 'Beam_Breast')
    plot_score_population(lookout_mean_population[outlier_inds], lookout_mean_population[inlier_inds], 'LookOut_Breast')