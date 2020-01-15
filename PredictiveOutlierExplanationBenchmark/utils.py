import json
import os
import collections


def calculate_metrics_avg(results_json):
    metrics_avg = {}
    reps = len(results_json.keys())
    for rep_id, res in results_json.items():
        for metric_key, val in res.items():
            for conf_id in val:
                if metric_key not in metrics_avg:
                    metrics_avg[metric_key] = {}
                if conf_id not in metrics_avg[metric_key]:
                    metrics_avg[metric_key][conf_id] = 0
                metrics_avg[metric_key][conf_id] += val[conf_id]['performance']
    # Take the average
    for metric_id, conf in metrics_avg.items():
        for conf_id, mval in conf.items():
            metrics_avg[metric_id][conf_id] = mval / reps
    return metrics_avg


def get_results_files(dir_path, starting=None, ending=None, exclude=None, contains=None):
    fileslist = []
    assert os.path.isdir(dir_path)
    allfiles = os.listdir(dir_path)
    for f in allfiles:
        start_cond = True if starting is None else f.startswith(starting)
        end_cond = True if ending is None else f.endswith(ending)
        contains_cond = True if contains is None else contains in f
        exclude_cond = True if exclude is None else exclude not in f
        if start_cond and end_cond and contains_cond and exclude_cond:
            fileslist.append(os.path.join(dir_path, f))
    return fileslist


def get_results_as_dict(results_files):
    metric_dim_results_dict = {}
    for f in results_files:
        with open(f) as json_file:
            results = json.load(json_file)
            datasetname = (os.path.splitext(os.path.basename(f))[0]).split('_')[0]
            dim = int(datasetname.replace("hics", "").strip())
            metrics_results_avg = calculate_metrics_avg(results)
            for metric_key, results in metrics_results_avg.items():
                if metric_key not in metric_dim_results_dict:
                    metric_dim_results_dict[metric_key] = {}
                for alg, perfomance in results.items():
                    if alg not in metric_dim_results_dict[metric_key]:
                        metric_dim_results_dict[metric_key][alg] = {}
                    if dim not in metric_dim_results_dict[metric_key][alg]:
                        metric_dim_results_dict[metric_key][alg][dim] = perfomance
                metric_dim_results_dict[metric_key] = \
                    dict(collections.OrderedDict(sorted(metric_dim_results_dict[metric_key].items())))
    for metric_key, algs_dict in metric_dim_results_dict.items():
        for alg, performances in algs_dict.items():
            metric_dim_results_dict[metric_key][alg] = dict(collections.OrderedDict(sorted(performances.items())))
    return metric_dim_results_dict


def feature_count_df(dir_path):
    print(get_results_files(dir_path, contains='log'))
