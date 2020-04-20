import argparse
from pathlib import Path
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
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


def __best_model_dict(benchmark_dict, fs):
    best_model_per_metric = {}
    for rep, rep_data in benchmark_dict.items():
        for m_id, metric_data in rep_data.items():
            best_model_per_metric.setdefault(m_id, {})
            for fsel_clf_id, conf_data in metric_data.items():
                if fs is True and 'none' in fsel_clf_id:
                    continue
                if fs is False and 'none' not in fsel_clf_id:
                    continue
                if len(best_model_per_metric[m_id]) == 0:
                    best_model_per_metric[m_id] = conf_data
                if best_model_per_metric[m_id]['effectiveness'] < conf_data['effectiveness']:
                    best_model_per_metric[m_id] = conf_data
                elif conf_data['effectiveness'] == best_model_per_metric[m_id]['effectiveness']:
                    if len(conf_data['feature_selection']['features']) < len(best_model_per_metric[m_id]['feature_selection']['features']):
                        best_model_per_metric[m_id] = conf_data
    # Refine the key names in dict
    for m_id, best_conf in best_model_per_metric.items():
        fsel_clf_id = best_conf['feature_selection']['id'] + '_' + best_conf['classifiers']['id']
        best_model_per_metric[m_id] = {fsel_clf_id: best_conf}
        print(m_id, fsel_clf_id, best_conf['effectiveness'])
    return best_model_per_metric


if __name__ == '__main__':
    global results
    results = read_benchmark_results('results/classification/datasets/synthetic/hics/group_g1/hics_20_g1/pseudo_samples_0')
    __best_model_dict(results, False)
    __best_model_dict(results, True)
    # analyze_metric('f1_score_outliers')
