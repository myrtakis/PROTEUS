import pandas as pd
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)
from models import Classifier
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
    return 'results_predictive_grouping/' + detector + \
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
    print('AUC on test', results['fs']['roc_auc']['hold_out_effectiveness'])
    return expl_features_ids, best_conf_id


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


def anomaly_threshold(detector, dataset, expl_features):
    Xtest, Ytest = get_X_Y(detector, dataset, 'pseudo_samples_0_hold_out_data.csv', expl_features, pseudo_samples=0)
    return len(np.where(Ytest == 1)[0])


def quality_assessment(expl_size, detector, dataset, ps_samples):
    results_path = get_base_path(detector, dataset, ps_samples) + '/expl_size_' + expl_size + '/best_model.json'
    expl_features, best_conf_id = get_explanation_features(results_path)
    clf_id, clf_params = get_clf_from_json(results_path)
    Xtrain, Ytrain = get_X_Y(detector, dataset, 'pseudo_samples_' + str(ps_samples) + '_data.csv', expl_features, ps_samples)
    Xtest, Ytest = get_X_Y(detector, dataset, 'pseudo_samples_' + str(ps_samples) + '_hold_out_data.csv', expl_features, ps_samples)
    predictions, auc = build_model(Xtrain, Ytrain, Xtest, Ytest, clf_id, clf_params)
    fda, fdn = detector_conflicts_ground_truth(get_Ytrue(dataset, detector), Ytest)
    tea, ten, fen, fea = get_interesting_points(predictions, Ytest, anomaly_threshold(detector, dataset, expl_features))
    fda_fen = len(fda.intersection(fen)) / len(fen) if len(fen) > 0 else -1
    fdn_fea = len(fdn.intersection(fea)) / len(fea) if len(fea) > 0 else -1
    return fda_fen, fdn_fea


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
    expl_size = '4'
    #ps_samples = 10
    fda_fen_dict = {}
    fdn_fea_dict = {}
    dataset_names = {'wbc_006': 'Breast Cancer', 'arrhythmia_015': 'Arrhythmia', 'ionosphere_006': 'Ionosphere'}
    for ps_samples in [0, 3, 10]:
        for dataset in ['wbc_006', 'arrhythmia_015', 'ionosphere_006']:
            for detector in ['iforest', 'lof', 'loda']:
                print(dataset, detector, ps_samples)
                key = dataset_names[dataset] + ' ' + detector
                fda_fen_dict.setdefault(key, {})
                fdn_fea_dict.setdefault(key, {})
                fda_fen, fdn_fea = quality_assessment(expl_size, detector, dataset, ps_samples)
                fdn_fea_dict[key][ps_samples] = fdn_fea
                fda_fen_dict[key][ps_samples] = fda_fen
                # fdn_fea_dict[dataset][detector] = fdn_fea
                # fda_fen_dict[dataset][detector] = fda_fen
    fda_fen_df = prepare_df(fda_fen_dict)
    fdn_fea_df = prepare_df(fdn_fea_dict)

    plot_df(fda_fen_df, 'FDA $\cap$ FEN')
    plot_df(fdn_fea_df, 'FDN $\cap$ FEA')
    # fda_fen_df = pd.DataFrame(fda_fen_dict).T
    # fdn_fea_df = pd.DataFrame(fdn_fea_dict).T
    print()