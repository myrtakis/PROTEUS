from PredictiveOutlierExplanationBenchmark.src.baselines.micencova import CA_Lasso
import pandas as pd
from PredictiveOutlierExplanationBenchmark.src.models.detectors import *
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.baselines.shapley import SHAP
import numpy as np


def run_micencova(dataset):
    cal = CA_Lasso(dataset)
    cal.run()


def run_shap(dataset, detector, params):
    detector.train(dataset.get_X().iloc[dataset.get_inlier_indices(), :], params)
    shap = SHAP(dataset, detector.predict_scores)
    local_explanations = shap.run()
    global_explanation = local_explanations.mean(axis=0)
    sorted_global_expl = np.argsort(global_explanation)[::-1]
    print(local_explanations)


def load_loda_explanation(dir_path):
    pass


if __name__ == '__main__':
    df = pd.read_csv('../results_normal/random_oversampling/lof/classification/datasets/synthetic/hics/group_g1/hics_100_g1_001/pseudo_samples_0/pseudo_samples_0_data.csv')
    # df = pd.read_csv('../datasets/synthetic/hics/group_g1/hics_20_g1.csv')
    dataset = Dataset(df, 'is_anomaly', 'subspaces')
    # run_micencova(dataset)
    run_shap(dataset, Lof(), {'n_neighbors':15})
