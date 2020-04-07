from sklearn.metrics import roc_auc_score, recall_score, precision_score
import numpy as np


_ROC_AUC = 'roc_auc'
_PRECISION_OUTLIERS = 'precision_outliers'
_RECALL_OUTLIERS = 'recall_outliers'


def metric_names():
    return [_ROC_AUC, _PRECISION_OUTLIERS, _RECALL_OUTLIERS]


def calculate_all_metrics(y_true, y_pred):
    return {
        **calculate_roc_auc(y_true, y_pred),
        **calculate_precision_outliers(y_true, y_pred),
        **calculate_recall_outliers(y_true, y_pred)
    }


def calculate_metric(y_true, y_pred, metric_id):
    if metric_id == _ROC_AUC:
        return calculate_roc_auc(y_true, y_pred)
    elif metric_id == _PRECISION_OUTLIERS:
        return calculate_precision_outliers(y_true, y_pred)
    elif metric_id == _RECALL_OUTLIERS:
        return calculate_recall_outliers(y_true, y_pred)
    else:
        assert False, 'Metric ' + metric_id + ' not found'


def calculate_roc_auc(y_true, y_pred):
    return {_ROC_AUC: roc_auc_score(y_true, y_pred)}


def calculate_precision_outliers(y_true, y_pred):
    if np.count_nonzero(y_pred == 1) == 0:
        return {_PRECISION_OUTLIERS: 0}
    else:
        return {_PRECISION_OUTLIERS: precision_score(y_true, y_pred, pos_label=1)}


def calculate_recall_outliers(y_true, y_pred):
    return {_RECALL_OUTLIERS: recall_score(y_true, y_pred, pos_label=1)}

