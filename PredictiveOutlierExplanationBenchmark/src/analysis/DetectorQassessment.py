import pandas as pd
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)
from models import Classifier
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import json


def get_X_Y(detector, dataset, df_name, expl_features, pseudo_samples):
    df = pd.read_csv(get_base_path(detector, dataset, pseudo_samples) + '/' + df_name)
    X = df.iloc[:, expl_features]
    Y = df.loc[:, 'is_anomaly']
    return X, Y


def get_base_path(detector, dataset, pseudo_samples):
    return get_dataset_path(detector, dataset) + '/pseudo_samples_' + \
           str(pseudo_samples) + '/noise_0'


def get_dataset_path(detector, dataset):
    return 'D:/' + 'results_predictive_grouping/' + detector + \
           '/protean/random_oversampling/classification/datasets/real/' + dataset


def get_clf_from_json(results_path):
    with open(results_path) as f:
        results = json.load(f)
    return results['fs']['roc_auc']['classifiers']['id'], results['fs']['roc_auc']['classifiers']['params']


def get_explanation_features(results_path):
    with open(results_path) as f:
        results = json.load(f)
    expl_features_ids = results['fs']['roc_auc']['feature_selection']['features']
    best_conf_id = results['fs']['roc_auc']['id']
    return expl_features_ids, best_conf_id, results['fs']['roc_auc']['hold_out_effectiveness']


def build_model(Xtrain, Ytrain, Xtest, Ytest, clf_id, clf_params):
    clf = Classifier()
    clf.setup_classifier_manually(clf_id, clf_params)
    predictions = clf.train(Xtrain, Ytrain).predict_proba(Xtest)
    return predictions, roc_auc_score(Ytest, predictions)


def get_Ytrue(dataset, detector):
    with open(get_dataset_path(detector, dataset) + '/train_holdout_inds.json') as f:
        holdout_inds = json.load(f)['holdout_inds']
    original_dataset = pd.read_csv('datasets/real/' + dataset.split('_')[0] + '.csv')
    return original_dataset.loc[holdout_inds, 'is_anomaly']


def detector_conflicts_ground_truth(Ytrue, Ydet):
    assert len(Ytrue) == len(Ydet), str(len(Ytrue)) + ' == ' + str(len(Ydet))
    Ytrue = Ytrue.values
    Ydet = Ydet.values
    true_anomalies = np.where(Ytrue == 1)[0]
    true_normals = np.where(Ytrue == 0)[0]
    detected_anomalies = np.where(Ydet == 1)[0]
    detected_normals = np.where(Ydet == 0)[0]
    fp = set(true_normals).intersection(detected_anomalies)
    fn = set(true_anomalies).intersection(detected_normals)
    return fp, fn


def get_interesting_points(predictions, Y, anomaly_threshold):
    sorted_pids = np.argsort(predictions)[::-1]
    proteus_anom = sorted_pids[:anomaly_threshold + 1]
    proteus_normals = sorted_pids[anomaly_threshold + 1:]
    given_anom = np.where(Y == 1)[0]
    given_normals = np.where(Y == 0)[0]
    suspicious_normals = set(given_normals).intersection(proteus_anom)
    suspicious_anomalies = set(given_anom).intersection(proteus_normals)
    trustworthy_anomalies = set(given_anom).intersection(proteus_anom)
    trustworthy_normals = set(given_normals).intersection(proteus_normals)
    return trustworthy_anomalies, trustworthy_normals, suspicious_anomalies, suspicious_normals


def anomaly_threshold(detector, dataset, expl_features, ps_samples):
    Xtest, Ytest = get_X_Y(detector, dataset, 'pseudo_samples_' + str(ps_samples) + '_hold_out_data.csv', expl_features, ps_samples)
    return len(np.where(Ytest == 1)[0])


def plot_sns(conflict_df, anomaly_thresholds_dict, title):
    ax = sns.stripplot(data=conflict_df, x='comb', y='pred', hue='true')
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        text = text.get_text()
        threshold_score = anomaly_thresholds_dict[text]
        ax.plot([tick - 0.2, tick + 0.2], [threshold_score, threshold_score], color='k')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.ylabel('log(Predictions)')
    plt.xlabel('Combinations')
    plt.show()


def build_pred_conflicts_df(predictions, Ytrue, detector, dataset, conflicted_points):
    dataset_names = {'wbc_006': 'Breast Cancer', 'arrhythmia_015': 'Arrhythmia', 'ionosphere_006': 'Ionosphere'}
    comb = np.full(len(predictions), dataset_names[dataset] + ' ' + detector)
    pred_df = pd.DataFrame(np.array([predictions, Ytrue, comb]).T, columns=['pred', 'true', 'comb'])
    pred_df = pred_df.astype({'pred': 'float16'})
    pred_df = pred_df.astype({'true': 'int8'})
    return pred_df.loc[conflicted_points]


def quality_assessment(expl_size, detector, dataset, ps_samples):
    results_path = get_base_path(detector, dataset, ps_samples) + '/expl_size_' + expl_size + '/best_model.json'
    expl_features, best_conf_id, auc_test = get_explanation_features(results_path)
    clf_id, clf_params = get_clf_from_json(results_path)
    Xtrain, Ytrain = get_X_Y(detector, dataset, 'pseudo_samples_' + str(ps_samples) + '_data.csv', expl_features, ps_samples)
    Xtest, Ytest = get_X_Y(detector, dataset, 'pseudo_samples_' + str(ps_samples) + '_hold_out_data.csv', expl_features, ps_samples)
    predictions, auc = build_model(Xtrain, Ytrain, Xtest, Ytest, clf_id, clf_params)
    threshold = anomaly_threshold(detector, dataset, expl_features, ps_samples)
    threshold_val = (sorted(predictions)[threshold] + sorted(predictions)[threshold+1]) / 2
    fda, fdn = detector_conflicts_ground_truth(get_Ytrue(dataset, detector), Ytest)
    tea, ten, fen, fea = get_interesting_points(predictions, Ytest, threshold)
    fda_fen = len(fda.intersection(fen)) / len(fen) if len(fen) > 0 else -1
    fdn_fea = len(fdn.intersection(fea)) / len(fea) if len(fea) > 0 else -1
    pred_fea_df = build_pred_conflicts_df(predictions, Ytest, detector, dataset, fea)
    pred_fen_df = build_pred_conflicts_df(predictions, Ytest, detector, dataset, fen)
    return {'fda_fen':fda_fen, 'fdn_fea':fdn_fea, 'fen':fen, 'fea':fea, 'anom_threshold': threshold_val,
            'auc_test':auc_test, 'pred_fea_df': pred_fea_df, 'pred_fen_df': pred_fen_df}


def prepare_df(data_dict):
    data_df = pd.DataFrame(data_dict).T
    data_df = data_df.astype(dict.fromkeys(data_df.columns, 'float16'))
    data_df = data_df.replace(dict.fromkeys(data_df.columns, -1), np.nan)
    return data_df


def plot_df(df, title):
    markers = ['s', 'o', '*']
    for i, c in enumerate(df.columns):
        plt.plot(df.index, df[c], label=str(c) + ' pseudo samples', marker=markers[i])
    plt.legend()
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()



if __name__ == '__main__':
    expl_size = '6'

    fda_fen_dict = {}
    fdn_fea_dict = {}
    auc_test_dict = {}
    fen_card_dict = {}
    fea_card_dict = {}
    anom_thresholds_dict = {}
    pred_fea_df = pd.DataFrame()
    pred_fen_df = pd.DataFrame()
    dataset_names = {'wbc_006': 'Breast Cancer', 'arrhythmia_015': 'Arrhythmia', 'ionosphere_006': 'Ionosphere'}
    for ps_samples in [10]:
        for dataset in ['wbc_006', 'arrhythmia_015', 'ionosphere_006']:
            for detector in ['iforest', 'lof', 'loda']:
                #print(dataset, detector, ps_samples)
                key = dataset_names[dataset] + ' ' + detector
                fda_fen_dict.setdefault(key, {})
                fdn_fea_dict.setdefault(key, {})
                fen_card_dict.setdefault(key, {})
                fea_card_dict.setdefault(key, {})
                auc_test_dict.setdefault(key, {})
                results = quality_assessment(expl_size, detector, dataset, ps_samples)
                fdn_fea_dict[key][ps_samples] = results['fdn_fea']
                fda_fen_dict[key][ps_samples] = results['fda_fen']
                auc_test_dict[key][ps_samples] = results['auc_test']
                fen_card_dict[key][ps_samples] = len(results['fen'])
                fea_card_dict[key][ps_samples] = len(results['fea'])
                pred_fea_df = pred_fea_df.append(results['pred_fea_df'], ignore_index=True)
                pred_fen_df = pred_fen_df.append(results['pred_fen_df'], ignore_index=True)
                anom_thresholds_dict[key] = results['anom_threshold']
                # fdn_fea_dict[dataset][detector] = fdn_fea
                # fda_fen_dict[dataset][detector] = fda_fen
    fda_fen_df = prepare_df(fda_fen_dict)
    fdn_fea_df = prepare_df(fdn_fea_dict)
    fen_card_df = prepare_df(fen_card_dict)
    fea_card_df = prepare_df(fea_card_dict)
    auc_test_df = prepare_df(auc_test_dict)

    # plot_df(fda_fen_df, 'FDA $\cap$ FEN')
    # plot_df(fdn_fea_df, 'FDN $\cap$ FEA')
    # plot_df(fen_card_df, '|FEN|')
    # plot_df(fea_card_df, '|FEA|')
    # plot_df(auc_test_df, 'AUC on Test')

    plot_sns(pred_fea_df, anom_thresholds_dict, 'False Explained Anomalies')
    plot_sns(pred_fen_df, anom_thresholds_dict, 'False Explained Negatives')

    # fda_fen_df = pd.DataFrame(fda_fen_dict).T
    # fdn_fea_df = pd.DataFrame(fdn_fea_dict).T
    print()