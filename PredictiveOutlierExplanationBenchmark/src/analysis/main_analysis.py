import argparse
from pathlib import Path
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from PredictiveOutlierExplanationBenchmark.src.utils import utils
import json


def read_benchmark_results(path_to_results_dir):
    bench_results = Path(path_to_results_dir, FileNames.best_models_bench_fname)
    with open(bench_results) as json_file:
        return json.load(json_file)


def analyze_familly():
    pass


def analyze_metric(metric_id):
    familly_data_in_metric = {}
    for rep_id, rep_data in results.items():
        for conf_id, conf_data in rep_data[metric_id].items():
            familly_data_in_metric.setdefault(conf_id, {})
            familly_data_in_metric[conf_id][rep_id] = conf_data['effectiveness']
    print(familly_data_in_metric)


if __name__ == '__main__':
    hics_p = '../results/classification/datasets/synthetic/hics/group_g1/hics_100_g1'
    with open(Path(hics_p, FileNames.navigator_fname)) as json_file:
        nav_file = json.load(json_file)
    global results
    results = read_benchmark_results(nav_file[FileKeys.navigator_pseudo_samples_key]['pseudo_samples_0']['dir'])
    # print(utils.best_model_dict(results, False))
    # print(utils.best_model_dict(results, True))
    print(utils.optimal_pseudo_samples_per_metric(nav_file[FileKeys.navigator_pseudo_samples_key], True))
    # analyze_metric('f1_score_outliers')
