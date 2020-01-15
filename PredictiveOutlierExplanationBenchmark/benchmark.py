from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import json
from var_selection import run_var_selection
from classifiers import train_classifier
from classifiers import test_classifier
from metrics import run_metrics
import itertools
import time
import os
from pathlib import Path


def run_benchmark(args):
    with open(args.config) as json_file:
        conf = json.load(json_file)
        settings = conf['settings']
        for attr, dataset_conf in conf['datasets'].items():
            sss = StratifiedShuffleSplit(n_splits=settings['repetitions'], test_size=settings['test_size'])
            X, Y = get_X_Y(dataset_conf)
            runs_dict = {}
            runs_logs = {}
            tmp_runs_logs = {}
            counter = 1
            for train_index, test_index in sss.split(X, Y):
                start_time = time.time()
                print('\nRep = ', counter)
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
                best_models_trained, logs = run_cv(X_train, Y_train, conf['metrics'], conf['variable_selection'],
                                                   conf['classifiers'], settings['k'])
                best_models_tested, logs = test_best_trained_models(X_test, Y_test, best_models_trained,
                                                                    conf['metrics'], logs)
                runs_dict[counter] = best_models_tested
                tmp_runs_logs[counter] = logs
                counter += 1
                elapsed_time = time.time() - start_time
                print('\n%0.4f' % elapsed_time, 'sec')

            runs_logs['repetitions'] = tmp_runs_logs
            runs_logs['config'] = args.config
            logs_output_file = os.path.splitext(args.save_output)[0] + '_log.json'

            create_dir_if_not_exists(args.save_output)
            with open(args.save_output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(runs_dict, indent=4, separators=(',', ': '), ensure_ascii=False))
            with open(logs_output_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(runs_logs, indent=4, separators=(',', ': '), ensure_ascii=False))


def run_cv(X, Y, metrics_conf, var_selection_conf, classifier_conf, k):
    k = min(k, get_rarest_class_count(Y))
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)
    var_selection_kv, classifiers_kv = generate_param_combs(var_selection_conf, classifier_conf)
    conf_performances = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        conf_counter = 0
        for var_sel_conf_tuple in var_selection_kv:
            sel_vars = run_var_selection(var_sel_conf_tuple['alg_id'], var_sel_conf_tuple['params'], X_train, Y_train)
            for classifier_conf_tuple in classifiers_kv:
                print('\r', var_sel_conf_tuple, classifier_conf_tuple, end="")
                predictions_array = None
                if len(sel_vars) > 0:
                    model = train_classifier(classifier_conf_tuple['alg_id'], classifier_conf_tuple['params'],
                                             sel_vars, X_train, Y_train)
                    predictions_array = test_classifier(classifier_conf_tuple['alg_id'], model, sel_vars, X_test)
                metrics_values = calculate_metrics_values(predictions_array, metrics_conf, Y_test)
                conf_performances = update_conf_performance(conf_performances, conf_counter, metrics_values,
                                                            var_sel_conf_tuple, classifier_conf_tuple)
                conf_counter += 1
    best_models = select_best_models(compute_avg_performances(conf_performances, k))
    return train_best_models(X, Y, best_models)


def get_X_Y(dataset_conf):
    df = pd.read_csv(dataset_conf['dataset_path'])
    if 'subspaces' in df.columns:
        df = df.drop(columns='subspaces')
    Y = df[dataset_conf['target']]
    X = df.drop(columns=dataset_conf['target'])
    return X, Y


def get_rarest_class_count(Y):
    Y_list = Y.to_list()
    rarest_class_count = None
    for c in set(Y_list):
        c_occur = Y_list.count(c)
        if rarest_class_count is None or c_occur < rarest_class_count:
            rarest_class_count = c_occur
    return rarest_class_count


def generate_param_combs(var_selection_conf, classifier_conf):
    var_sel_conf_combs = []
    classifiers_conf_combs = []
    for var_sel_id in var_selection_conf:
        var_sel_params_keys = list(var_selection_conf[var_sel_id]['params'].keys())
        var_sel_params_vals = var_selection_conf[var_sel_id]['params'].values()
        params_combs = list(itertools.product(*var_sel_params_vals))
        var_sel_conf_combs.extend(build_key_value_params(var_sel_params_keys, params_combs, var_sel_id))

    for classifier_id in classifier_conf:
        classifier_params_keys = list(classifier_conf[classifier_id]['params'].keys())
        classifier_params_vals = classifier_conf[classifier_id]['params'].values()
        params_combs = list(itertools.product(*classifier_params_vals))
        omit_combs = None
        if 'omit_combinations' in classifier_conf[classifier_id]:
            omit_combs = classifier_conf[classifier_id]['omit_combinations']
        classifiers_conf_combs.extend(build_key_value_params(classifier_params_keys, params_combs,
                                                             classifier_id, omit_combs))
    return var_sel_conf_combs, classifiers_conf_combs


def omit_configuration(params, omit_combs):
    omit = True
    na = 'na'
    if omit_combs is None:
        return not omit

    omitted_combs = {}

    for item in omit_combs:
        for pparam, v1 in item['prime_param'].items():
            if str(params[pparam]).lower() != str(v1).lower():
                continue
            omitted_combs = item['combs']
            for oparam in item['combs']:
                if str(params[oparam]).lower() != na:
                    return omit

    for param, val in params.items():
        if str(val).lower() == na and param not in omitted_combs:
            return omit

    return not omit


def build_key_value_params(params_keys, params_combs, alg_id, omit_combs=None):
    params_kv = []
    for t in params_combs:
        kv = {}
        counter = 0
        for v in t:
            kv[params_keys[counter]] = v
            counter += 1
        if not omit_configuration(kv, omit_combs):
            params_kv.append({'alg_id': alg_id, 'params': kv})
    return params_kv


def calculate_metrics_values(predictions_array, metrics_conf, Y_test):
    metric_values = None

    if predictions_array is None:
        metric_values = {}
        for k in metrics_conf:
            metric_values[k] = 0
        return metric_values

    for p in predictions_array:
        tmp_metric_values = run_metrics(metrics_conf, Y_test, p)
        if metric_values is None:
            metric_values = tmp_metric_values
        else:
            for k in metric_values.keys():
                metric_values[k] += tmp_metric_values[k]

    for k in metric_values.keys():
        metric_values[k] = metric_values[k] / len(predictions_array)

    return metric_values


def update_conf_performance(conf_perfomances, conf_counter, metric_values, var_sel_tuple, classifier_tuple):
    if len(conf_perfomances) < conf_counter + 1:
        conf_perfomances.append({'var_sel': var_sel_tuple, 'classifier': classifier_tuple, 'metrics': metric_values})
    else:
        for m in conf_perfomances[conf_counter]['metrics']:
            conf_perfomances[conf_counter]['metrics'][m] += metric_values[m]
    return conf_perfomances


def compute_avg_performances(conf_performances, k):
    for i in range(len(conf_performances)):
        for m in conf_performances[i]['metrics']:
            conf_performances[i]['metrics'][m] = conf_performances[i]['metrics'][m]/k
    return conf_performances


def select_best_models(conf_performances):
    best_models_perfomances = {}
    for e in conf_performances:
        conf_key = e['var_sel']['alg_id'] + '_' + e['classifier']['alg_id']
        for metric_key in e['metrics']:
            if metric_key not in best_models_perfomances:
                best_models_perfomances[metric_key] = {}
            if conf_key not in best_models_perfomances[metric_key]:
                best_models_perfomances[metric_key][conf_key] = {'performance': e['metrics'][metric_key],
                                                                 'var_sel': e['var_sel'], 'classifier': e['classifier']}
            if best_models_perfomances[metric_key][conf_key]['performance'] < e['metrics'][metric_key]:
                best_models_perfomances[metric_key][conf_key] = {'performance': e['metrics'][metric_key],
                                                                 'var_sel': e['var_sel'],
                                                                 'classifier': e['classifier']}
    return best_models_perfomances


def train_best_models(X_train, Y_train, best_models):
    best_models_trained = {}
    logs = {}
    for metric_key in best_models.keys():
        for conf_key in best_models[metric_key].keys():
            var_sel_start_time = time.time()
            sel_features = run_var_selection(best_models[metric_key][conf_key]['var_sel']['alg_id'],
                                             best_models[metric_key][conf_key]['var_sel']['params'], X_train, Y_train)
            var_sel_elapsed = time.time() - var_sel_start_time
            train_model_start_time = time.time()
            model = train_classifier(best_models[metric_key][conf_key]['classifier']['alg_id'],
                                     best_models[metric_key][conf_key]['classifier']['params'],
                                     sel_features, X_train, Y_train)
            train_model_elapsed_time = time.time() - train_model_start_time
            if metric_key not in best_models_trained:
                best_models_trained[metric_key] = {}
            best_models_trained[metric_key][conf_key] = {'model': model, 'sel_features': sel_features,
                                                         'var_sel': best_models[metric_key][conf_key]['var_sel'],
                                                         'classifier': best_models[metric_key][conf_key]['classifier']}
            if metric_key not in logs:
                logs[metric_key] = {}
            if conf_key not in logs[metric_key]:
                logs[metric_key][conf_key] = {'var_sel_elapsed_time': var_sel_elapsed, 'train_model_elapsed_time': train_model_elapsed_time}
    return best_models_trained, logs


def test_best_trained_models(X_test, Y_test, best_models_trained, metrics_conf, logs):
    best_models_test_performances = {}
    for metric_key in best_models_trained.keys():
        for conf_key in best_models_trained[metric_key].keys():
            sel_features = best_models_trained[metric_key][conf_key]['sel_features']
            predictions_array = test_classifier(best_models_trained[metric_key][conf_key]['classifier']['alg_id'],
                                                best_models_trained[metric_key][conf_key]['model'], sel_features, X_test)
            metrics_values = calculate_metrics_values(predictions_array, metrics_conf, Y_test)
            if metric_key not in best_models_test_performances:
                best_models_test_performances[metric_key] = {}
            if 'none' in conf_key:
                sel_features = 'all'
            else:
                sel_features = str(sel_features)
            best_models_test_performances[metric_key][conf_key] = {'performance': metrics_values[metric_key],
                                                                   'var_sel': best_models_trained[metric_key][conf_key]['var_sel'],
                                                                   'classifier': best_models_trained[metric_key][conf_key]['classifier'],
                                                                   'sel_features': sel_features}
            logs[metric_key][conf_key]['tested_indexes'] = list(X_test.index)
            logs[metric_key][conf_key]['true_labels'] = Y_test.tolist()
            logs[metric_key][conf_key]['predictions'] = predictions_array[0].tolist()
    return best_models_test_performances, logs


def create_dir_if_not_exists(save_dir):
    path = Path(save_dir)
    parent = path.parent
    if not os.path.exists(parent):
        os.makedirs(parent)
