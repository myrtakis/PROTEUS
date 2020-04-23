import json
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
from pathlib import Path


def optimal_pseudo_samples_per_metric(paths_to_pseudo_samples_data, fs):
    ps_dict = get_pseudo_samples_bench_results(paths_to_pseudo_samples_data)
    best_model_per_metric = {}
    for k, bench_results in ps_dict.items():
        curr_best_model = best_model_dict(bench_results, fs)
        for m_id, data in curr_best_model.items():
            best_model_per_metric.setdefault(m_id, None)
            if best_model_per_metric[m_id] is None or best_model_per_metric[m_id]['best_model']['effectiveness'] < data['effectiveness']:
                best_model_per_metric[m_id] = {'bench_results': bench_results, 'best_model': data, 'k': k}
    return best_model_per_metric


def get_pseudo_samples_bench_results(paths_to_pseudo_samples_data):
    ps_dict = {}
    for k, data in paths_to_pseudo_samples_data.items():
        dir_to_ps_data = data[FileKeys.navigator_pseudo_sample_dir_key]
        with open(Path(dir_to_ps_data, FileNames.best_models_bench_fname)) as json_file:
            pseudo_samples_bench = json.load(json_file)
            ps_dict[data[FileKeys.navigator_pseudo_samples_num_key]] = pseudo_samples_bench
    return ps_dict


def best_model_dict(benchmark_dict, fs, metric=None):
    best_model_per_metric = {}
    avg_perf = avg_performance_per_family(benchmark_dict)
    avg_features_selected = avg_num_features_selected(benchmark_dict)
    for m_id, metric_data in avg_perf.items():
        if metric is not None and m_id != metric:
            continue
        best_model_per_metric.setdefault(m_id, None)
        for fsel_clf_id, effectiveness in metric_data.items():
            if fs is True and 'none' in fsel_clf_id:
                continue
            if fs is False and 'none' not in fsel_clf_id:
                continue
            if best_model_per_metric[m_id] is None:
                best_model_per_metric[m_id] = {'effectiveness': effectiveness, 'conf_id': fsel_clf_id}
            if best_model_per_metric[m_id]['effectiveness'] < effectiveness:
                best_model_per_metric[m_id] = {'effectiveness': effectiveness, 'conf_id': fsel_clf_id}
            elif best_model_per_metric[m_id]['effectiveness'] == effectiveness:
                best_conf_id = best_model_per_metric[m_id]['conf_id']
                if avg_features_selected[m_id][best_conf_id] > avg_features_selected[m_id][fsel_clf_id]:
                    best_model_per_metric[m_id] = {'effectiveness': effectiveness, 'conf_id': fsel_clf_id}
    return best_model_per_metric


def best_conf_in_repetitions(benchmark_dict, conf_id, metric=None):
    best_model_per_metric = {}
    for rep, rep_data in benchmark_dict.items():
        for m_id, metric_data in rep_data.items():
            if metric is not None and m_id != metric:
                continue
            best_model_per_metric.setdefault(m_id, {})
            for fsel_clf_id, conf_data in metric_data.items():
                if conf_id != fsel_clf_id:
                    continue
                if len(best_model_per_metric[m_id]) == 0:
                    best_model_per_metric[m_id] = conf_data
                if best_model_per_metric[m_id]['effectiveness'] < conf_data['effectiveness']:
                    best_model_per_metric[m_id] = conf_data
                elif conf_data['effectiveness'] == best_model_per_metric[m_id]['effectiveness']:
                    if len(conf_data['feature_selection']['features']) < len(
                            best_model_per_metric[m_id]['feature_selection']['features']):
                        best_model_per_metric[m_id] = conf_data
    # Refine the key names in dict
    for m_id, best_conf in best_model_per_metric.items():
        fsel_clf_id = best_conf['feature_selection']['id'] + '_' + best_conf['classifiers']['id']
        best_model_per_metric[m_id] = {fsel_clf_id: best_conf}
    return best_model_per_metric


def avg_performance_per_family(benchmark_dict):
    avg_performance = {}
    for rep, rep_data in benchmark_dict.items():
        for m_id, metric_data in rep_data.items():
            avg_performance.setdefault(m_id, {})
            for fsel_clf_id, conf_data in metric_data.items():
                avg_performance[m_id].setdefault(fsel_clf_id, 0.0)
                avg_performance[m_id][fsel_clf_id] += conf_data['effectiveness']
    reps = len(benchmark_dict.keys())
    for m_id, data in avg_performance.items():
        for fsel_clf_id, family_data in data.items():
            avg_performance[m_id][fsel_clf_id] /= reps
    return avg_performance


def avg_num_features_selected(benchmark_dict):
    avg_num_features = {}
    for rep, rep_data in benchmark_dict.items():
        for m_id, metric_data in rep_data.items():
            avg_num_features.setdefault(m_id, {})
            for fsel_clf_id, conf_data in metric_data.items():
                avg_num_features[m_id].setdefault(fsel_clf_id, 0)
                avg_num_features[m_id][fsel_clf_id] += len(conf_data['feature_selection']['features'])
    reps = len(benchmark_dict.keys())
    for m_id, data in avg_num_features.items():
        for fsel_clf_id, family_data in data.items():
            avg_num_features[m_id][fsel_clf_id] /= reps
    return avg_num_features
