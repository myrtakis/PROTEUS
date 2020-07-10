import argparse
from configpkg import *
from holders.Dataset import *
from pipeline.NormalPipeline import NormalPipeline
from pipeline.PredictivePipeline import PredictivePipeline
import pandas as pd
import numpy as np

RUN_NORMAL_PIPELINE = False
OVERSAMPLING_METHOD = 'random'


def run_pipeline(config_file_path, detector, save_dir):
    ConfigMger.setup_configs(config_file_path)
    dataset_pd, original_dims = keep_relevant_features()
    original_dataset = Dataset(dataset_pd, DatasetConfig.get_anomaly_column_name(),
                               DatasetConfig.get_subspace_column_name())
    if RUN_NORMAL_PIPELINE:
        NormalPipeline(save_dir, original_dataset, detector).run()
    else:
        PredictivePipeline(save_dir, original_dataset, OVERSAMPLING_METHOD, original_dims, detector).run()


def keep_relevant_features():
    dataset_pd = pd.read_csv(DatasetConfig.get_dataset_path())
    original_dims = None if DatasetConfig.get_subspace_column_name() is None else dataset_pd.shape[1]
    if DatasetConfig.get_subspace_column_name() is not None:
        ground_truth_df = pd.concat([
            dataset_pd.loc[:, DatasetConfig.get_anomaly_column_name()],
            dataset_pd.loc[:, DatasetConfig.get_subspace_column_name()]],
            axis=1)
        rel_features = set()
        subspaces = dataset_pd.loc[:, DatasetConfig.get_subspace_column_name()]
        for sub in set(subspaces[np.where(subspaces != '-')[0]]):
            sub_as_set = set(map(int, sub[sub.index('[') + 1: sub.index(']')].split()))
            rel_features = rel_features.union(sub_as_set)
        dataset_pd = pd.concat([dataset_pd.iloc[:, list(rel_features)], ground_truth_df], axis=1)
    return dataset_pd, original_dims


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file path.', required=True)
    parser.add_argument('-d', '--detector', help='Define the detector', default=None)
    parser.add_argument('-sd', '--save_dir', help='The directory to save to old_results', default='old_results')
    args = parser.parse_args()

    run_pipeline(args.config, args.detector, args.save_dir)
