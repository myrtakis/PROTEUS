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


def run_benchmark(config_path):
    rep = 25
    with open(config_path) as json_file:
        conf = json.load(json_file)
        for attr, dataset_conf in conf['datasets'].items():
            sss = StratifiedShuffleSplit(n_splits=rep, test_size=0.3, random_state=0)
            X, Y = get_X_Y(dataset_conf)
            runs_dict = {}
            counter = 0
            for train_index, test_index in sss.split(X, Y):
                print('\nRep = ', counter)
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
                best_models_trained = run_cv(X_train, Y_train, conf['metrics'], conf['variable_selection'], conf['classifiers'])
                best_models_tested = test_best_trained_models(X_test, Y_test, best_models_trained, conf['metrics'])
                runs_dict[counter] = best_models_tested
                counter += 1
            with open('results.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(runs_dict, indent=4, separators=(',', ': '), ensure_ascii=False))


def run_cv(X, Y, metrics_conf, var_selection_conf, classifier_conf):
    k = 5
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)
    var_selection_kv, classifiers_kv = generate_param_combs(var_selection_conf, classifier_conf)
    conf_performances = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        conf_counter = 0
        fold_counter = 0
        for var_sel_conf_tuple in var_selection_kv:
            sel_vars = run_var_selection(var_sel_conf_tuple['alg_id'], var_sel_conf_tuple['params'], X_train, Y_train)
            for classifier_conf_tuple in classifiers_kv:
                print('\rFold =', (fold_counter % k) + 1, var_sel_conf_tuple, classifier_conf_tuple, end="")
                fold_counter += 1
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
    Y = df[dataset_conf['target']]
    X = df.drop(columns=dataset_conf['target'])
    return X, Y


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
        classifiers_conf_combs.extend(build_key_value_params(classifier_params_keys, params_combs, classifier_id))

    return var_sel_conf_combs, classifiers_conf_combs


def build_key_value_params(params_keys, params_combs, alg_id):
    params_kv = []
    for t in params_combs:
        kv = {}
        counter = 0
        for v in t:
            kv[params_keys[counter]] = v
            counter += 1
        params_kv.append({'alg_id': alg_id, 'params': kv})
    return params_kv


def calculate_metrics_values(predictions_array, metrics_conf, Y_test):
    metric_values = None
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
    best_models_perfomance = {}
    for e in conf_performances:
        key = e['var_sel']['alg_id'] + '_' + e['classifier']['alg_id']
        if key not in best_models_perfomance:
            best_models_perfomance[key] = {'metrics': e['metrics'], 'var_sel': e['var_sel'],
                                           'classifier': e['classifier']}
        for m in e['metrics']:
            best_performance = max(e['metrics'][m], best_models_perfomance[key]['metrics'][m])
            best_models_perfomance[key]['metrics'][m] = best_performance
    return best_models_perfomance


def train_best_models(X_train, Y_train, best_models):
    best_models_trained = {}
    for mkey in best_models.keys():
        sel_features = run_var_selection(best_models[mkey]['var_sel']['alg_id'],
                                         best_models[mkey]['var_sel']['params'], X_train, Y_train)
        model = train_classifier(best_models[mkey]['classifier']['alg_id'],
                                 best_models[mkey]['classifier']['params'],
                                 sel_features, X_train, Y_train)
        best_models_trained[mkey] = {'model': model, 'sel_features': sel_features,
                                     'var_sel': best_models[mkey]['var_sel'],
                                     'classifier': best_models[mkey]['classifier']}
    return best_models_trained


def test_best_trained_models(X_test, Y_test, best_models_trained, metrics_conf):
    best_models_test_performances = {}
    for mkey in best_models_trained:
        sel_features = best_models_trained[mkey]['sel_features']
        predictions_array = test_classifier(best_models_trained[mkey]['classifier']['alg_id'],
                                            best_models_trained[mkey]['model'], sel_features, X_test)
        metrics_values = calculate_metrics_values(predictions_array, metrics_conf, Y_test)
        best_models_test_performances[mkey] = {'metrics': metrics_values,
                                               'var_sel': best_models_trained[mkey]['var_sel'],
                                               'classifier': best_models_trained[mkey]['classifier']}
    return best_models_test_performances
