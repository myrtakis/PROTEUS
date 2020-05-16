from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import *
from PredictiveOutlierExplanationBenchmark.src.pipeline.NormalPipeline import NormalPipeline
from PredictiveOutlierExplanationBenchmark.src.pipeline.PredictivePipeline import PredictivePipeline


class Pipeline:

    __RUN_NORMAL_PIPELINE = False

    def __init__(self):
        pass

    @staticmethod
    def run(config_file_path, save_dir):
        ConfigMger.setup_configs(config_file_path)
        original_dataset = Dataset(DatasetConfig.get_dataset_path(), DatasetConfig.get_anomaly_column_name(),
                                   DatasetConfig.get_subspace_column_name())

        if Pipeline.__RUN_NORMAL_PIPELINE:
            NormalPipeline(save_dir, original_dataset, oversampling_method='random').run()
        else:
            PredictivePipeline(save_dir, original_dataset, oversampling_method='random').run()
