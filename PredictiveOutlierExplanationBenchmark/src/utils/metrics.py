from sklearn.metrics import roc_auc_score, recall_score, precision_score
import numpy as np


_ROC_AUC = 'roc_auc'
_PRECISION_OUTLIERS = 'precision_outliers'
_RECALL_OUTLIERS = 'recall_outliers'
_F1_SCORE_OUTLIERS = 'f1_score_outliers'


def metric_names():
    return [_ROC_AUC, _PRECISION_OUTLIERS, _RECALL_OUTLIERS, _F1_SCORE_OUTLIERS]


def calculate_all_metrics(y_true, y_pred):
    perfs = {}
    for m in metric_names():
        perfs.update(calculate_metric(y_true, y_pred, m))
    return perfs


def calculate_metric(y_true, y_pred, metric_id):
    if metric_id == _ROC_AUC:
        return calculate_roc_auc(y_true, y_pred)
    elif metric_id == _PRECISION_OUTLIERS:
        return calculate_precision_outliers(y_true, y_pred)
    elif metric_id == _RECALL_OUTLIERS:
        return calculate_recall_outliers(y_true, y_pred)
    elif metric_id == _F1_SCORE_OUTLIERS:
        return calculate_f1_score_outliers(y_true, y_pred)
    else:
        assert False, 'Metric ' + metric_id + ' not found'


def calculate_roc_auc(y_true, y_pred):
    return {_ROC_AUC: roc_auc_score(y_true, y_pred)}


def calculate_precision_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    if np.count_nonzero(y_pred == 1) == 0:
        return {_PRECISION_OUTLIERS: 0.0}
    else:
        return {_PRECISION_OUTLIERS: precision_score(y_true, y_pred, pos_label=1)}


def calculate_recall_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    return {_RECALL_OUTLIERS: recall_score(y_true, y_pred, pos_label=1)}


def calculate_f1_score_outliers(y_true, y_pred):
    y_pred = make_y_pred_is_binary(y_pred)
    recall = recall_score(y_true, y_pred, pos_label=1)
    if np.count_nonzero(y_pred == 1) == 0:
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, pos_label=1)
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
