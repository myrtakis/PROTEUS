from PredictiveOutlierExplanationBenchmark.src.models.OutlierDetector import Detector
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.configpkg.SettingsConfig import SettingsConfig
from math import floor
import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_all_metrics


def detect_outliers(dataset):
    detector = Detector()
    detector.train(dataset.get_X())
    scores = detector.score_samples()
    if SettingsConfig.is_classification_task():
        new_dataset, threshold = create_dataset_classification(SettingsConfig.get_top_k_points_to_explain(), dataset, scores)
        detector.set_labels(dataset.get_Y())
        detector.set_effectiveness(assess_detector(new_dataset.get_Y(), dataset.get_Y()))
        return new_dataset, detector, threshold
    else:
        new_dataset, threshold = create_dataset_regression(dataset, scores)
        return new_dataset, detector, threshold


def create_dataset_classification(top_k_points, dataset, scores):
    top_k_points_to_keep = floor(top_k_points * dataset.get_X().shape[0])
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
    return calculate_all_metrics(y_true, y_pred)
