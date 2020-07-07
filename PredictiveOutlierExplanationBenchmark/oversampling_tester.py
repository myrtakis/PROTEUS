import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
src = os.path.join(currentdir, 'src')
sys.path.insert(0, src)

from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
import numpy as np

from PredictiveOutlierExplanationBenchmark.src.models.OutlierDetector import Detector
from PredictiveOutlierExplanationBenchmark.src.models.detectors import iForest
from PredictiveOutlierExplanationBenchmark.src.models.detectors.Lof import Lof
from PredictiveOutlierExplanationBenchmark.src.pipeline.DatasetTransformer import Transformer


def read_datasets(data_dir, synthetic):
    datasets = {}
    for f in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, f)):
            continue
        dataset = Dataset(os.path.join(data_dir, f), 'is_anomaly', 'subspaces' if synthetic else None)
        datasets[dataset.get_X().shape[1]] = dataset
    return dict(sorted(datasets.items()))


def add_new_samples(datasets_detected, method, k):
    outlier_ratio_per_dataset = {}
    for dim, dataset_info in datasets_detected.items():
        dataset = dataset_info['dataset']
        threshold = dataset_info['threshold']
        detector = dataset_info['detector']
        augm_dataset = Transformer(method=method).transform(dataset, k, detector, threshold)
        ps_indices = np.arange(dataset.get_X().shape[0], augm_dataset.get_X().shape[0])
        ps_labels = augm_dataset.get_Y()[ps_indices]
        ps_outlier_ratio = list(ps_labels).count(1) / len(ps_labels)
        outlier_ratio_per_dataset[dim] = ps_outlier_ratio
    return outlier_ratio_per_dataset


def detect_outliers(datasets, synthetic):
    datasets_detected = {}
    for dim, dataset in datasets.items():
        detector = Detector(Lof(), 'lof', {'n_neighbors': 15})
        # detector = Detector(iForest(), 'iforest', {'n_estimators': 100, 'max_samples': 256})
        detector.train(dataset.get_X())
        scores = detector.get_scores_in_train()
        outlier_ratio = 0.01 # float(len(dataset.get_outlier_indices())) / float(dataset.get_X().shape[0])
        topk = int(np.floor(len(scores) * outlier_ratio))
        threshold = sorted(scores, reverse=True)[topk]
        labels = np.zeros(len(scores), dtype=int)
        labels[np.where(scores > threshold)[0]] = 1
        new_df = dataset.get_df()
        new_df['is_anomaly'] = labels
        new_dataset = Dataset(new_df, 'is_anomaly', 'subspaces' if synthetic else None)
        datasets_detected[dim] = {'dataset': new_dataset, 'threshold': threshold, 'detector': detector}
    return datasets_detected


if __name__ == '__main__':
    data, synth = 'datasets/synthetic/hics/group_g1', True
    # data, synth = 'datasets/real', False
    datasets = read_datasets(data, synth)
    datasets_detected = detect_outliers(datasets, synth)
    oversampling_size = [10]
    for k in oversampling_size:
        for method in ['random']:
            outlier_ratios = add_new_samples(datasets_detected, method, k)
            print(k, method, outlier_ratios)
