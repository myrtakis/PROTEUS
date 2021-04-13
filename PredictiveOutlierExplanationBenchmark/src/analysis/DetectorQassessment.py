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


def detector_auc(detector, dataset):
    with open(get_dataset_path(detector, dataset) + '/' + 'detectors_info.json') as f:
        det_info = json.load(f)
        train_auc = det_info[detector]['effectiveness']['roc_auc']
        test_auc = det_info[detector]['hold_out_effectiveness']['roc_auc']
        return {'AUC train': train_auc, 'AUC test': test_auc}


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
    plt.xticks(rotation=40)
    plt.yscale('log')
    plt.ylabel('log(Predictions)')
    plt.xlabel('Combinations')
    plt.tight_layout()
    plt.show()


def build_pred_conflicts_df(predictions, Ytrue, detector, dataset, conflicted_points):
    dataset_names = {'wbc_006': 'Breast Cancer', 'arrhythmia_015': 'Arrhythmia', 'ionosphere_006': 'Ionosphere'}
    comb = np.full(len(predictions), dataset_names[dataset] + '\n' + detector)
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
    threshold_val = (sorted(predictions, reverse=True)[threshold] + sorted(predictions, reverse=True)[threshold+1]) / 2
    Ytrue_test = get_Ytrue(dataset, detector).reset_index(drop=True).astype('int32')
    fda, fdn = detector_conflicts_ground_truth(Ytrue_test, Ytest)
    tea, ten, fen, fea = get_interesting_points(predictions, Ytest, threshold)
    fda_fen = len(fda.intersection(fen)) / len(fen) if len(fen) > 0 else -1
    fdn_fea = len(fdn.intersection(fea)) / len(fea) if len(fea) > 0 else -1
    pred_fea_df = build_pred_conflicts_df(predictions, Ytrue_test, detector, dataset, fea)
    pred_fen_df = build_pred_conflicts_df(predictions, Ytrue_test, detector, dataset, fen)
    return {'fda_fen':fda_fen, 'fdn_fea':fdn_fea, 'fen':fen, 'fea':fea, 'anom_threshold': threshold_val,
            'auc_test':auc_test, 'pred_fea_df': pred_fea_df, 'pred_fen_df': pred_fen_df}


def prepare_df(data_dict):
    data_df = pd.DataFrame(data_dict).T
    data_df = data_df.astype(dict.fromkeys(data_df.columns, 'float16'))
    data_df = data_df.replace(dict.fromkeys(data_df.columns, -1), np.nan)
    return data_df


def plot_df(df, ylabel, title):
    markers = ['s', 'o', '*']
    for i, c in enumerate(df.columns):
        plt.plot(df.index, df[c], label=str(c) + ' pseudo samples', marker=markers[i])
    plt.legend()
    plt.xticks(rotation=35)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_auc_det(det_auc_df):
    plt.plot(det_auc_df.index, det_auc_df['AUC train'], label='AUC train', marker='o')
    plt.plot(det_auc_df.index, det_auc_df['AUC test'], label='AUC test', marker='o')
    plt.xticks(rotation=35)
    plt.legend()
    plt.title('Detectors AUCs')
    plt.show()


def plot_info_of_best_aucs(auc_test_df, fda_fen_df, fdn_fea_df):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    best_ps_fda_fen = {}
    best_ps_fdn_fea = {}
    best_auc_vals = {}
    for i in best_ps_df.index:
        best_ps_fdn_fea[i] = fdn_fea_df.loc[i, best_ps_df.loc[i]]
        best_ps_fda_fen[i] = fda_fen_df.loc[i, best_ps_df.loc[i]]
        best_auc_vals[i] = auc_test_df.loc[i, best_ps_df.loc[i]]
    plt.plot(list(best_ps_fdn_fea.keys()), list(best_ps_fdn_fea.values()), label='False Negatives Discovery', marker='o')
    plt.plot(list(best_ps_fda_fen.keys()), list(best_ps_fda_fen.values()), label='False Positives Discovery', marker='o')
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()

    plt.plot(list(best_auc_vals.keys()), list(best_auc_vals.values()), marker='o')
    plt.title('Best PROTEUS AUCs on Test')
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()


def plot_best_score_distr(auc_test_df, pred_fea_dict, pred_fen_dict,  anom_thresholds_dict):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    best_pred_fea= pd.DataFrame()
    best_pred_fen = pd.DataFrame()
    for i in best_ps_df.index:
        best_ps = best_ps_df.loc[i]
        best_pred_fea = best_pred_fea.append(pred_fea_dict[best_ps][pred_fea_dict[best_ps]['comb'] == i], ignore_index=True)
        best_pred_fen = best_pred_fen.append(pred_fen_dict[best_ps][pred_fen_dict[best_ps]['comb'] == i], ignore_index=True)
    plot_sns(best_pred_fea, anom_thresholds_dict, 'False Explained Anomalies ps_samples=' + str(ps_samples))
    plot_sns(best_pred_fen, anom_thresholds_dict, 'False Explained Normals ps_samples=' + str(ps_samples))


if __name__ == '__main__':
    expl_size = '10'

    fda_fen_dict = {}
    fdn_fea_dict = {}
    auc_test_dict = {}
    fen_card_dict = {}
    fea_card_dict = {}
    det_auc_dict = {}
    anom_thresholds_dict = {}
    pred_fea_dict = {}
    pred_fen_dict = {}
    dataset_names = {'wbc_006': 'Breast Cancer', 'arrhythmia_015': 'Arrhythmia', 'ionosphere_006': 'Ionosphere'}
    for ps_samples in [0,3,10]:
        for dataset in ['wbc_006', 'arrhythmia_015', 'ionosphere_006']:
            for detector in ['iforest', 'lof', 'loda']:
                #print(dataset, detector, ps_samples)
                key = dataset_names[dataset] + ' ' + detector
                key = key[:key.rindex(' ')] + '\n' + key[key.rindex(' ') + 1:]
                fda_fen_dict.setdefault(key, {})
                fdn_fea_dict.setdefault(key, {})
                fen_card_dict.setdefault(key, {})
                fea_card_dict.setdefault(key, {})
                auc_test_dict.setdefault(key, {})
                det_auc_dict.setdefault(key, {})
                pred_fea_dict.setdefault(ps_samples, pd.DataFrame())
                pred_fen_dict.setdefault(ps_samples, pd.DataFrame())
                results = quality_assessment(expl_size, detector, dataset, ps_samples)
                fdn_fea_dict[key][ps_samples] = results['fdn_fea']
                fda_fen_dict[key][ps_samples] = results['fda_fen']
                auc_test_dict[key][ps_samples] = results['auc_test']
                fen_card_dict[key][ps_samples] = len(results['fen'])
                fea_card_dict[key][ps_samples] = len(results['fea'])
                det_auc_dict[key] = detector_auc(detector, dataset)
                pred_fea_dict[ps_samples] = pred_fea_dict[ps_samples].append(results['pred_fea_df'], ignore_index=True)
                pred_fen_dict[ps_samples] = pred_fen_dict[ps_samples].append(results['pred_fen_df'], ignore_index=True)
                anom_thresholds_dict[key] = results['anom_threshold']
                # fdn_fea_dict[dataset][detector] = fdn_fea
                # fda_fen_dict[dataset][detector] = fda_fen
    fda_fen_df = prepare_df(fda_fen_dict)
    fdn_fea_df = prepare_df(fdn_fea_dict)
    fen_card_df = prepare_df(fen_card_dict)
    fea_card_df = prepare_df(fea_card_dict)
    auc_test_df = prepare_df(auc_test_dict)

    # plot_auc_det(pd.DataFrame(det_auc_dict).T)

    # plot_info_of_best_aucs(auc_test_df, fda_fen_df, fdn_fea_df)
    plot_best_score_distr(auc_test_df, pred_fea_dict, pred_fen_dict, anom_thresholds_dict)

    # plot_df(fda_fen_df, '(FDA $\cap$ FEN) / |FEN|', 'False Positives Discovery')
    # plot_df(fdn_fea_df, '(FDN $\cap$ FEA) / |FEA|', 'Fasle Negatives Discovery')
    # plot_df(fen_card_df, '|FEN|', 'Number of Conflicts (cases 10)')
    # plot_df(fea_card_df, '|FEA|', 'Number of Conflicts (cases 01)')
    # plot_df(auc_test_df, 'AUC', 'PROTEUS AUC on Test')



    print()