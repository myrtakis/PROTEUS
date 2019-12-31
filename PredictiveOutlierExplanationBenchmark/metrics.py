from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def run_metrics(metrics_conf, Y_true, Y_pred):
    metrics_map = {
        "roc_auc": run_roc_auc,
        "avep": run_average_precision
    }
    metrics_values = {}
    for metric in metrics_conf:
        metrics_values[metric] = metrics_map[metric](Y_true, Y_pred)
    return metrics_values


def run_roc_auc(Y_true, Y_pred):
    return roc_auc_score(Y_true, Y_pred)


def run_average_precision(Y_true, Y_pred):
    return average_precision_score(Y_true, Y_pred)
