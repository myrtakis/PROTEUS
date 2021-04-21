import pandas as pd
import os, sys, inspect

from sklearn import manifold
from sklearn.decomposition import PCA

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(currentdir)
sys.path.insert(0, grandpadir)
from utils.spider_plotting import construct_spider_plot
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
        return {'Detector AUC on train set': train_auc, 'AUC AUC on test': test_auc}


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


def biuld_points_label_set(detector, tea, ten, fea, fen, fda, fdn):
    false_negatives = fdn.intersection(fea)
    false_positives = fda.intersection(fen)
    tea = tea.union(fen.symmetric_difference(fda))
    ten = ten.union(fea.symmetric_difference(fdn))
    return merge_dfs(detector, tea, ten, false_positives, false_negatives)


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
            'auc_test':auc_test, 'pred_fea_df': pred_fea_df, 'pred_fen_df': pred_fen_df,
            'all_points_df': biuld_points_label_set(detector, tea, ten, fea, fen, fda, fdn), 'explanation': expl_features}


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


def plot_aucs(det_auc_df, columns_to_plot):
    # for c in columns_to_plot:
    #     plt.plot(det_auc_df.index, det_auc_df[c], label=c, marker='o')
    det_auc_df.loc[:, columns_to_plot].plot.bar()
    plt.xticks(rotation=35)
    plt.xlabel('Combinations')
    plt.ylabel('AUC')
    #plt.ylim([0,1])
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_dir = 'figures/quality_assessment'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'QAaucComp.png'), dpi=300)
    plt.clf()


def plot_info_of_best_aucs(auc_test_df, fda_fen_df, fdn_fea_df):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    best_ps_fda_fen = {}
    best_ps_fdn_fea = {}
    for i in best_ps_df.index:
        best_ps_fdn_fea[i] = fdn_fea_df.loc[i, best_ps_df.loc[i]]
        best_ps_fda_fen[i] = fda_fen_df.loc[i, best_ps_df.loc[i]]
    # plt.plot(list(best_ps_fdn_fea.keys()), list(best_ps_fdn_fea.values()), label='FND', marker='o')
    # plt.plot(list(best_ps_fda_fen.keys()), list(best_ps_fda_fen.values()), label='FPD', marker='o')
    joined_df = pd.DataFrame([best_ps_fdn_fea.values(), best_ps_fda_fen.values()]).T
    joined_df.index = best_ps_fdn_fea.keys()
    joined_df.columns = ['TND', 'TAD']
    joined_df.plot.bar()
    plt.xticks(rotation=35)
    plt.yticks(np.arange(0, 1.1, .1), [str(int(i * 100)) + '%' for i in np.arange(0, 1.1, .1)])
    plt.xlabel('Combinations')
    plt.ylabel('Discoveries (%)')
    plt.legend()
    plt.tight_layout()
    save_dir = 'figures/quality_assessment'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'QAdiscoveries.png'), dpi=300)
    plt.clf()


def plot_best_score_distr(auc_test_df, pred_fea_dict, pred_fen_dict,  anom_thresholds_dict):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    best_pred_fea = pd.DataFrame()
    best_pred_fen = pd.DataFrame()
    best_anom_thresholds = {}
    for i in best_ps_df.index:
        best_ps = best_ps_df.loc[i]
        best_pred_fea = best_pred_fea.append(pred_fea_dict[best_ps][pred_fea_dict[best_ps]['comb'] == i], ignore_index=True)
        best_pred_fen = best_pred_fen.append(pred_fen_dict[best_ps][pred_fen_dict[best_ps]['comb'] == i], ignore_index=True)
        best_anom_thresholds[i] = anom_thresholds_dict[best_ps][i]
    plot_sns(best_pred_fea, best_anom_thresholds, 'False Explained Anomalies (cases 01)')
    plot_sns(best_pred_fen, best_anom_thresholds, 'False Explained Normals (cases 10)')


def proteus_best_aucs_test_set(auc_test_df):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    best_auc_vals = {}
    for i in best_ps_df.index:
        best_auc_vals[i] = auc_test_df.loc[i, best_ps_df.loc[i]]
    return best_auc_vals


def merge_dfs(detector, tea, ten, false_positives, false_negatives):
    # df of form [(tea, 'tea'), (ten ,'ten')...]
    tea_df = pd.DataFrame(np.array([list(tea), np.full(len(tea), detector + ' Anomaly')]).T, columns=['ids', 'label']).astype({'ids': 'int32'})
    ten_df = pd.DataFrame(np.array([list(ten), np.full(len(ten), detector + ' Normal')]).T, columns=['ids', 'label']).astype({'ids': 'int32'})
    false_positives_df = pd.DataFrame(np.array([list(false_positives), np.full(len(false_positives), detector + ' True Normal')]).T, columns=['ids', 'label']).astype({'ids': 'int32'})
    false_negatives_df = pd.DataFrame(np.array([list(false_negatives), np.full(len(false_negatives), detector + ' True Anomaly')]).T, columns=['ids', 'label']).astype({'ids': 'int32'})
    return pd.concat([tea_df, ten_df, false_negatives_df, false_positives_df])


def plot_fp_fn_points(auc_test_df, all_points_dict, explanation_per_comb):
    dim_reduction_methods = {
        'pca':
            {
                'method': PCA(n_components=2, random_state=0),
                'axis_labels': ['Principal Component 1', 'Principal Component 2']
            },
        'tsne':
            {
                'method': manifold.TSNE(n_components=2, init='pca', perplexity=40, random_state=0),
                'axis_labels': ['t-SNE Embedding 1', 't-SNE Embedding 2']
            }
    }
    best_ps_df = auc_test_df.idxmax(axis='columns')
    points_of_best_configs = {}
    best_explanations_dict = {}
    dataset_names = {'Breast Cancer': 'wbc_006', 'Arrhythmia':'arrhythmia_015', 'Ionosphere':'ionosphere_006'}
    for i in best_ps_df.index:
        best_ps = best_ps_df.loc[i]
        points_of_best_configs[i] = all_points_dict[i][best_ps]
        best_explanations_dict[i] = explanation_per_comb[i][best_ps]
    for c in points_of_best_configs:
        dataset, detector = c.split('\n')[0], c.split('\n')[1]
        if not (dataset == 'Arrhythmia' and detector == 'loda'):
            continue
        Xtest, _ = get_X_Y(detector, dataset_names[dataset], 'pseudo_samples_' + str(best_ps_df.loc[c]) + '_hold_out_data.csv',
                               best_explanations_dict[c], best_ps_df.loc[c])
        alg = 'tsne'
        reduced_df = pd.DataFrame(dim_reduction_methods[alg]['method'].fit_transform(Xtest), columns=dim_reduction_methods[alg]['axis_labels'])
        reduced_df['label'] = np.full(reduced_df.shape[0], '')
        reduced_df.loc[points_of_best_configs[c]['ids'].values, 'label'] = points_of_best_configs[c]['label'].values
        plot_2d_scatter(reduced_df, dataset + ' ' + detector, detector, ['Normal', 'Anomaly', 'True Normal'])
        plot_2d_scatter(reduced_df, dataset + ' ' + detector, detector, ['Normal', 'Anomaly', 'True Anomaly'])
    pass


def plot_2d_scatter(df, plot_title, detector, labels):
    markers = {'Normal': 'o', 'Anomaly': 'o', 'True Normal': 's', 'True Anomaly': '^'}
    msizes = {'Normal': 40, 'Anomaly': 40, 'True Normal': 80, 'True Anomaly': 80}
    colors = {'Normal': 'skyblue', 'Anomaly': 'tomato', 'True Normal': 'black', 'True Anomaly': 'black'}
    if len(np.unique(df['label'])) != 4:
        return
    cols = list(set(df.columns).difference({'label'}))
    for l in labels:
        key = detector + ' ' + l
        if 'True' not in l:
            key_name = detector + ' ' + l + ', PROTEUS ' + l
        elif 'Normal' in l:
            key_name = detector + ' Anomaly' + ', PROTEUS Normal'
        else:
            key_name = detector + ' Normal' + ', PROTEUS Anomaly'
        x = df[df['label'] == key].loc[:, cols[0]]
        y = df[df['label'] == key].loc[:, cols[1]]
        plt.scatter(x, y, marker=markers[l], color=colors[l], label=key_name, s=msizes[l])
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.legend()
    # plt.title(plot_title)
    save_dir = 'figures/quality_assessment'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_name = 'tsneANC' if 'True Normal' in labels else 'tsneNAC'
    plt.tight_layout()
    plt.savefig(save_dir + '/' + file_name + '.png', dpi=300)
    plt.clf()


def points_demo(auc_test_df, all_points_dict, explanation_dict, feature_names):
    best_ps_df = auc_test_df.idxmax(axis='columns')
    points_of_best_configs = {}
    best_explanations_dict = {}
    dataset_names = {'Breast Cancer': 'wbc_006', 'Arrhythmia': 'arrhythmia_015', 'Ionosphere': 'ionosphere_006'}
    for i in best_ps_df.index:
        best_ps = best_ps_df.loc[i]
        points_of_best_configs[i] = all_points_dict[i][best_ps]
        best_explanations_dict[i] = explanation_dict[i][best_ps]
    for c in points_of_best_configs:
        dataset, detector = c.split('\n')[0], c.split('\n')[1]
        if not (dataset == 'Ionosphere' and detector == 'lof'):
            continue
        points_df = points_of_best_configs[c].reset_index(drop=True)
        Xtest, _ = get_X_Y(detector, dataset_names[dataset],
                           'pseudo_samples_' + str(best_ps_df.loc[c]) + '_hold_out_data.csv',
                           best_explanations_dict[c], best_ps_df.loc[c])
        feature_names[dataset_names[dataset]] = ['    ' + f + '    ' for f in feature_names[dataset_names[dataset]]]
        Xtest.columns = np.array(feature_names[dataset_names[dataset]])[best_explanations_dict[c]]
        Xtest = Xtest.loc[:, Xtest.nunique() > 3]
        interesting_points = find_interesting_points(Xtest, points_df)
        if len(interesting_points) > 0:
            construct_spider_plot(Xtest, interesting_points['agreements'], savedir='figures/quality_assessment', name='agreements_spider.png')
            construct_spider_plot(Xtest, interesting_points['conflicts'], savedir='figures/quality_assessment', name='conflicts_spider.png')


def find_interesting_points(data, points_df):
    df_groups = points_df.groupby('label')
    data.quantile(0.75) - data.quantile(0.25)
    best_points = {'conflicts':{}, 'agreements':{}}
    if len(df_groups.groups) != 4:
        return {}
    for g in df_groups.groups:
        data_of_group = data.iloc[points_df.loc[df_groups.groups[g], 'ids'], :]
        if 'Normal' in g:
            d = (data_of_group > data.quantile(0.25)) & (data_of_group < data.quantile(0.75))
            label = 'Anomaly Normal Conflict' if 'True' in g else 'Normal Normal Agreement'
        else:
            d = (data_of_group < data.quantile(0.25)) | (data_of_group > data.quantile(0.75))
            label = 'Normal Anomaly Conflict' if 'True' in g else 'Anomaly Anomaly Agreement'
        kind = 'conflicts' if 'True' in g else 'agreements'
        best_points[kind][d.sum(axis=1).argmax()] = label
    return best_points


if __name__ == '__main__':
    expl_size = '10'

    feature_names = {
        'wbc_006': ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'],
        'arrhythmia_015': ['age','sex','height','weight','QRSduration','PRinterval','Q-Tinterval','Tinterval','Pinterval','QRS','chDI_Qwave','chDI_Rwave','chDI_Swave','chDI_RPwave','chDI_intrinsicReflecttions','chDI_RRwaveExists','chDI_DD_RRwaveExists','chDI_RPwaveExists','chDI_DD_RPwaveExists','chDI_RTwaveExists','chDI_DD_RTwaveExists','chDII_Qwave','chDII_Rwave','chDII_Swave','chDII_RPwave','chDII_SPwave','chDII_intrinsicReflecttions','chDII_RRwaveExists','chDII_DD_RRwaveExists','chDII_RPwaveExists','chDII_DD_RPwaveExists','chDII_RTwaveExists','chDII_DD_RTwaveExists','chDIII_Qwave','chDIII_Rwave','chDIII_Swave','chDIII_RPwave','chDIII_SPwave','chDIII_intrinsicReflecttions','chDIII_RRwaveExists','chDIII_DD_RRwaveExists','chDIII_RPwaveExists','chDIII_DD_RPwaveExists','chDIII_RTwaveExists','chDIII_DD_RTwaveExists','chAVR_Qwave','chAVR_Rwave','chAVR_Swave','chAVR_RPwave','chAVR_SPwave','chAVR_intrinsicReflecttions','chAVR_RRwaveExists','chAVR_DD_RRwaveExists','chAVR_RPwaveExists','chAVR_DD_RPwaveExists','chAVR_RTwaveExists','chAVR_DD_RTwaveExists','chAVL_Qwave','chAVL_Rwave','chAVL_Swave','chAVL_RPwave','chAVL_intrinsicReflecttions','chAVL_DD_RRwaveExists','chAVL_RPwaveExists','chAVL_DD_RPwaveExists','chAVL_RTwaveExists','chAVL_DD_RTwaveExists','chAVF_Qwave','chAVF_Rwave','chAVF_Swave','chAVF_RPwave','chAVF_SPwave','chAVF_intrinsicReflecttions','chAVF_RRwaveExists','chAVF_DD_RRwaveExists','chAVF_DD_RPwaveExists','chAVF_RTwaveExists','chAVF_DD_RTwaveExists','chV1_Qwave','chV1_Rwave','chV1_Swave','chV1_RPwave','chV1_SPwave','chV1_intrinsicReflecttions','chV1_RRwaveExists','chV1_DD_RRwaveExists','chV1_RPwaveExists','chV1_DD_RPwaveExists','chV1_RTwaveExists','chV1_DD_RTwaveExists','chV2_Qwave','chV2_Rwave','chV2_Swave','chV2_RPwave','chV2_SPwave','chV2_intrinsicReflecttions','chV2_RRwaveExists','chV2_DD_RRwaveExists','chV2_RPwaveExists','chV2_DD_RPwaveExists','chV2_RTwaveExists','chV2_DD_RTwaveExists','chV3_Qwave','chV3_Rwave','chV3_Swave','chV3_RPwave','chV3_SPwave','chV3_intrinsicReflecttions','chV3_RRwaveExists','chV3_DD_RRwaveExists','chV3_RPwaveExists','chV3_DD_RPwaveExists','chV3_RTwaveExists','chV3_DD_RTwaveExists','chV4_Qwave','chV4_Rwave','chV4_Swave','chV4_RPwave','chV4_SPwave','chV4_intrinsicReflecttions','chV4_RRwaveExists','chV4_DD_RRwaveExists','chV4_RTwaveExists','chV4_DD_RTwaveExists','chV5_Qwave','chV5_Rwave','chV5_Swave','chV5_RPwave','chV5_intrinsicReflecttions','chV5_DD_RRwaveExists','chV5_DD_RPwaveExists','chV5_DD_RTwaveExists','chV6_Qwave','chV6_Rwave','chV6_Swave','chV6_RPwave','chV6_intrinsicReflecttions','chV6_RRwaveExists','chV6_DD_RRwaveExists','chV6_RPwaveExists','chV6_DD_RTwaveExists','chDI_JJwaveAmp','chDI_QwaveAmp','chDI_RwaveAmp','chDI_SwaveAmp','chDI_RPwaveAmp','chDI_PwaveAmp','chDI_TwaveAmp','chDI_QRSA','chDI_QRSTA','chDII_JJwaveAmp','chDII_QwaveAmp','chDII_RwaveAmp','chDII_SwaveAmp','chDII_RPwaveAmp','chDII_SPwaveAmp','chDII_PwaveAmp','chDII_TwaveAmp','chDII_QRSA','chDII_QRSTA','chDIII_JJwaveAmp','chDIII_QwaveAmp','chDIII_RwaveAmp','chDIII_SwaveAmp','chDIII_RPwaveAmp','chDIII_SPwaveAmp','chDIII_PwaveAmp','chDIII_TwaveAmp','chDIII_QRSA','chDIII_QRSTA','chAVR_JJwaveAmp','chAVR_QwaveAmp','chAVR_RwaveAmp','chAVR_SwaveAmp','chAVR_RPwaveAmp','chAVR_SPwaveAmp','chAVR_PwaveAmp','chAVR_TwaveAmp','chAVR_QRSA','chAVR_QRSTA','chAVL_JJwaveAmp','chAVL_QwaveAmp','chAVL_RwaveAmp','chAVL_SwaveAmp','chAVL_RPwaveAmp','chAVL_PwaveAmp','chAVL_TwaveAmp','chAVL_QRSA','chAVL_QRSTA','chAVF_JJwaveAmp','chAVF_QwaveAmp','chAVF_RwaveAmp','chAVF_SwaveAmp','chAVF_RPwaveAmp','chAVF_SPwaveAmp','chAVF_PwaveAmp','chAVF_TwaveAmp','chAVF_QRSA','chAVF_QRSTA','chV1_JJwaveAmp','chV1_QwaveAmp','chV1_RwaveAmp','chV1_SwaveAmp','chV1_RPwaveAmp','chV1_SPwaveAmp','chV1_PwaveAmp','chV1_TwaveAmp','chV1_QRSA','chV1_QRSTA','chV2_JJwaveAmp','chV2_QwaveAmp','chV2_RwaveAmp','chV2_SwaveAmp','chV2_RPwaveAmp','chV2_SPwaveAmp','chV2_PwaveAmp','chV2_TwaveAmp','chV2_QRSA','chV2_QRSTA','chV3_JJwaveAmp','chV3_QwaveAmp','chV3_RwaveAmp','chV3_SwaveAmp','chV3_RPwaveAmp','chV3_SPwaveAmp','chV3_PwaveAmp','chV3_TwaveAmp','chV3_QRSA','chV3_QRSTA','chV4_JJwaveAmp','chV4_QwaveAmp','chV4_RwaveAmp','chV4_SwaveAmp','chV4_RPwaveAmp','chV4_SPwaveAmp','chV4_PwaveAmp','chV4_TwaveAmp','chV4_QRSA','chV4_QRSTA','chV5_JJwaveAmp','chV5_QwaveAmp','chV5_RwaveAmp','chV5_SwaveAmp','chV5_RPwaveAmp','chV5_PwaveAmp','chV5_TwaveAmp','chV5_QRSA','chV5_QRSTA','chV6_JJwaveAmp','chV6_QwaveAmp','chV6_RwaveAmp','chV6_SwaveAmp','chV6_RPwaveAmp','chV6_PwaveAmp','chV6_TwaveAmp','chV6_QRSA','chV6_QRSTA'],
        'ionosphere_006': ['Radar 0', 'Radar 1', 'Radar 2', 'Radar 3', 'Radar 4', 'Radar 5', 'Radar 6', 'Radar 7', 'Radar 8', 'Radar 9', 'Radar 10', 'Radar 11', 'Radar 12', 'Radar 13', 'Radar 14', 'Radar 15', 'Radar 16', 'Radar 17', 'Radar 18', 'Radar 19', 'Radar 20', 'Radar 21', 'Radar 22', 'Radar 23', 'Radar 24', 'Radar 25', 'Radar 26', 'Radar 27', 'Radar 28', 'Radar 29', 'Radar 30', 'Radar 31', 'Radar 32']
    }

    fda_fen_dict = {}
    fdn_fea_dict = {}
    auc_test_dict = {}
    fen_card_dict = {}
    fea_card_dict = {}
    det_auc_dict = {}
    all_points_dict = {}
    explanation_dict = {}
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
                explanation_dict.setdefault(key, {})
                all_points_dict.setdefault(key, {})
                anom_thresholds_dict.setdefault(ps_samples, {})
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
                all_points_dict[key][ps_samples] = results['all_points_df']
                explanation_dict[key][ps_samples] = results['explanation']
                anom_thresholds_dict[ps_samples][key] = results['anom_threshold']
                # fdn_fea_dict[dataset][detector] = fdn_fea
                # fda_fen_dict[dataset][detector] = fda_fen
    fda_fen_df = prepare_df(fda_fen_dict)
    fdn_fea_df = prepare_df(fdn_fea_dict)
    fen_card_df = prepare_df(fen_card_dict)
    fea_card_df = prepare_df(fea_card_dict)
    auc_test_df = prepare_df(auc_test_dict)

    # points_demo(auc_test_df, all_points_dict, explanation_dict, feature_names)

    prot_aucs_test = proteus_best_aucs_test_set(auc_test_df)
    for comb in det_auc_dict:
        det_auc_dict[comb]['Proteus AUC on test set'] = prot_aucs_test[comb]
    plot_aucs(pd.DataFrame(det_auc_dict).T, ['Proteus AUC on test set', 'Detector AUC on train set'])

    plot_info_of_best_aucs(auc_test_df, fda_fen_df, fdn_fea_df)
    # plot_best_score_distr(auc_test_df, pred_fea_dict, pred_fen_dict, anom_thresholds_dict)

    # plot_fp_fn_points(auc_test_df, all_points_dict, explanation_dict)

    # plot_df(fda_fen_df, '(FDA $\cap$ FEN) / |FEN|', 'True Normals Discovery')
    # plot_df(fdn_fea_df, '(FDN $\cap$ FEA) / |FEA|', 'Fasle Negatives Discovery')
    # plot_df(fen_card_df, '|FEN|', 'Number of Conflicts (cases 10)')
    # plot_df(fea_card_df, '|FEA|', 'Number of Conflicts (cases 01)')
    # plot_df(auc_test_df, 'AUC', 'PROTEUS AUC on Test')



    print()