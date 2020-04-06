from PredictiveOutlierExplanationBenchmark.src.configpkg import *
from PredictiveOutlierExplanationBenchmark.src.models.Classifier import Classifier
from PredictiveOutlierExplanationBenchmark.src.models.FeatureSelection import FeatureSelection
import json
from pathlib import Path
import os


class ResultsWriter:

    __default_folder = '../results'

    def __init__(self, benchmark_dict, best_model_dict, pseudo_samples, dataset):
        self.__benchmark_dict = benchmark_dict
        self.__best_model_dict = best_model_dict
        self.__pseudo_samples = pseudo_samples
        self.__pseudo_samples_key = 'pseudo_samples_' + str(pseudo_samples)
        self.__dataset = dataset
        self.__base_dir = None
        self.__final_dir = None
        self.__generate_dir()

    def write(self):
        best_model_fname = 'best_model.json'
        best_models_benchmark = {}
        info_dict = {}
        for rep, rep_data in self.__benchmark_dict.items():
            pass

        self.__prepare_best_model()
        with open(os.path.join(self.__final_dir, best_model_fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__best_model_dict, indent=4, separators=(',', ': '), ensure_ascii=False))

    def __prepare_best_model(self):
        tmp_dict = {}
        for m_id, metric_data in self.__best_model_dict.items():
            tmp_dict.setdefault(m_id, {})
            fsel_dict = metric_data.get_fsel().to_dict()
            clf_dict = metric_data.get_clf().to_dict()
            del fsel_dict[FeatureSelection.TIME_KEY]
            fsel_dict[FeatureSelection.FEATURES_KEY] = [int(x) for x in fsel_dict[FeatureSelection.FEATURES_KEY]]
            print()
            fsel_dict[FeatureSelection.EQUIVALENT_FEATURES_KEY] = [[int(x) for x in lst] for lst in fsel_dict[FeatureSelection.EQUIVALENT_FEATURES_KEY]]
            del clf_dict[Classifier.TIME_KEY]
            del clf_dict[Classifier.PREDICTIONS_KEY]
            tmp_dict[m_id].update(
                **{'effectiveness': metric_data.get_effectiveness()},
                **{FeatureSelectionConfig.feature_selection_key(): fsel_dict},
                **{ClassifiersConfig.classifier_key(): clf_dict}
            )
        self.__best_model_dict = tmp_dict

    def __generate_dir(self):
        dataset_path = DatasetConfig.get_dataset_path()
        if dataset_path.startswith('..'):
            dataset_path = os.path.join(*Path(dataset_path).parts[1:])
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_path = dataset_path.replace(os.path.basename(dataset_path), '')
        self.__base_dir = Path().joinpath(
            ResultsWriter.__default_folder,
            SettingsConfig.get_task(),
            dataset_path,
            base_name,
            DetectorConfig.get_id())
        final_dir = os.path.join(self.__base_dir, self.__pseudo_samples_key)
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        self.__final_dir = final_dir

    def get_base_dir(self):
        return self.__base_dir

