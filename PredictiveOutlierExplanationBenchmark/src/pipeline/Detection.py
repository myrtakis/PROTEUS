from PredictiveOutlierExplanationBenchmark.src.models.OutlierDetector import Detector
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import SettingsConfig
from math import floor
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_roc_auc


def detect_outliers(dataset):
    detectors_arr = Detector.init_detectors()
    best_detector = select_best_detector(detectors_arr, dataset)
    if SettingsConfig.is_classification_task():
        new_dataset, threshold = create_dataset_classification(dataset, best_detector.get_scores_in_train())
        return new_dataset, best_detector, threshold
    else:
        new_dataset, threshold = create_dataset_regression(dataset, best_detector.get_scores_in_train())
        return new_dataset, best_detector, threshold


def select_best_detector(detectors_arr, dataset):
    best_detector = None
    max_perf = None
    for det in detectors_arr:
        det.train(dataset.get_X())
        scores = det.get_scores_in_train()
        perf = assess_detector(scores, dataset.get_Y())
        det.set_effectiveness(perf)
        perf_value = perf[next(iter(perf))]
        if max_perf is None or perf_value > max_perf:
            max_perf = perf_value
            best_detector = det
    print('Best detector:', best_detector.get_id(), 'with auc score', max_perf)
    return best_detector


def create_dataset_classification(dataset, scores):
    top_k_points = SettingsConfig.get_top_k_points_to_explain(),
    top_k_points_to_keep = floor(top_k_points[0] * dataset.get_X().shape[0])
    desc_ordered_indices = np.argsort(scores)[::-1]
    desc_ordered_outlier_indices = desc_ordered_indices[range(top_k_points_to_keep)]
    lowest_outlier_score = scores[desc_ordered_outlier_indices[-1]]
    highest_inlier_score = scores[desc_ordered_indices[top_k_points_to_keep]]
    threshold = (lowest_outlier_score + highest_inlier_score) / 2.0
    labels = np.zeros(dataset.get_X().shape[0])
    labels[desc_ordered_outlier_indices] = 1
    df = dataset.get_X().copy()
    df[dataset.get_anomaly_column_name()] = labels
    assert list(df[dataset.get_anomaly_column_name()]).count(1) == top_k_points_to_keep
    if dataset.get_subspace_column_name() is not None:
        df[dataset.get_subspace_column_name()] = dataset.get_subspaces()
    new_dataset = Dataset(df, dataset.get_anomaly_column_name(), dataset.get_subspace_column_name())
    return new_dataset, threshold


def create_dataset_regression(dataset, scores):
    df = dataset.get_X().copy()
    df[dataset.get_anomaly_column_name()] = scores
    if dataset.get_subspace_column_name() is not None:
        df[dataset.get_subspace_column_name()] = dataset.get_subspaces()
    new_dataset = Dataset(df, dataset.get_anomaly_column_name(), dataset.get_subspace_column_name())
    threshold = None
    return new_dataset, threshold


def assess_detector(y_pred, y_true):
    assert SettingsConfig.is_classification_task()
    return calculate_roc_auc(y_true, y_pred)
