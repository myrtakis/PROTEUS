import argparse
from configpkg import *
from holders.Dataset import *
from pipeline.NormalPipeline import NormalPipeline
from pipeline.PredictivePipeline import PredictivePipeline

RUN_NORMAL_PIPELINE = False
OVERSAMPLING_METHOD = 'random'


def run_pipeline(config_file_path, detector, save_dir):
    ConfigMger.setup_configs(config_file_path)
    original_dataset = Dataset(DatasetConfig.get_dataset_path(), DatasetConfig.get_anomaly_column_name(),
                               DatasetConfig.get_subspace_column_name())

    if RUN_NORMAL_PIPELINE:
        NormalPipeline(save_dir, original_dataset, detector).run()
    else:
        PredictivePipeline(save_dir, original_dataset, OVERSAMPLING_METHOD, detector).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file path.', required=True)
    parser.add_argument('-d', '--detector', help='Define the detector', default=None)
    parser.add_argument('-sd', '--save_dir', help='The directory to save to old_results', default='old_results')
    args = parser.parse_args()

    run_pipeline(args.config, args.detector, args.save_dir)
