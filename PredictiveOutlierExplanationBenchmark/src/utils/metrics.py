from sklearn.metrics import roc_auc_score, recall_score, precision_score
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


Rfast = importr('Rfast')


_ROC_AUC = 'roc_auc'
_PRECISION_OUTLIERS = 'precision_outliers'
_RECALL_OUTLIERS = 'recall_outliers'
_F1_SCORE_OUTLIERS = 'f1_score_outliers'


def metric_names():
    return [_ROC_AUC, _PRECISION_OUTLIERS, _RECALL_OUTLIERS, _F1_SCORE_OUTLIERS]


def calculate_all_metrics(y_true, y_pred, run_R=False):
    perfs = {}
    for m in metric_names():
        perfs.update(calculate_metric(y_true, y_pred, m, run_R))
    return perfs


def calculate_metric(y_true, y_pred, metric_id, run_R=False):
    if metric_id == _ROC_AUC:
        return calculate_roc_auc(y_true, y_pred, run_R)
    elif metric_id == _PRECISION_OUTLIERS:
        return calculate_precision_outliers(y_true, y_pred)
    elif metric_id == _RECALL_OUTLIERS:
        return calculate_recall_outliers(y_true, y_pred)
    elif metric_id == _F1_SCORE_OUTLIERS:
        return calculate_f1_score_outliers(y_true, y_pred)
    else:
        assert False, 'Metric ' + metric_id + ' not found'


def calculate_roc_auc(y_true, y_pred, run_R=False):
    assert y_true.shape[0] == y_pred.shape[0]
    assert len(y_true.shape) == 1, len(y_true)
    y_true = np.copy(y_true)
    if len(np.unique(y_true)) == 1:
        return {_ROC_AUC: -1}
    if run_R is True:
        pandas2ri.activate()
        y_true = ro.r.matrix(y_true, nrow=y_true.shape[0], ncol=1)
        if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
            y_pred = ro.r.matrix(y_pred, nrow=y_pred.shape[0], ncol=y_pred.shape[1])
            return {_ROC_AUC: Rfast.colaucs(y_true, y_pred)}
        else:
            y_pred = ro.r.matrix(y_pred, nrow=y_pred.shape[0], ncol=1)
            return {_ROC_AUC: Rfast.auc(y_true, y_pred)[0]}
    else:
        return {_ROC_AUC: roc_auc_score(y_true, y_pred)}


def calculate_precision_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    conf_mat = conf_matrix(y_true, y_pred)
    if np.count_nonzero(y_pred == 1) == 0:
        return {_PRECISION_OUTLIERS: 0.0}
    else:
        prec = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'])
        return {_PRECISION_OUTLIERS: prec}


def calculate_recall_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    conf_mat = conf_matrix(y_true, y_pred)
    rec = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fn'])
    return {_RECALL_OUTLIERS: rec}


def conf_matrix(y_true, y_pred):
    return {
        'tn': len(np.intersect1d(np.where(y_true == 0)[0], np.where(y_pred == 0)[0])),
        'fp': len(np.intersect1d(np.where(y_true == 0)[0], np.where(y_pred == 1)[0])),
        'fn': len(np.intersect1d(np.where(y_true == 1)[0], np.where(y_pred == 0)[0])),
        'tp': len(np.intersect1d(np.where(y_true == 1)[0], np.where(y_pred == 1)[0]))
    }


def calculate_f1_score_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    recall = calculate_recall_outliers(y_true, y_pred)[_RECALL_OUTLIERS]
    precision = calculate_precision_outliers(y_pred, y_pred)[_PRECISION_OUTLIERS]
    if recall == 0 and precision == 0:
        return {_F1_SCORE_OUTLIERS: 0.0}
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))
        return {_F1_SCORE_OUTLIERS: f1}


def make_y_pred_is_binary(y_pred):
    assert min(y_pred) >= 0.0 and max(y_pred) <= 1.0, 'min %f, max %f' % (min(y_pred), max(y_pred))
    y_pred_cp = np.copy(y_pred)
    y_pred_cp[np.where(y_pred >= 0.5)] = 1
    y_pred_cp[np.where(y_pred < 0.5)] = 0
    return y_pred_cp
