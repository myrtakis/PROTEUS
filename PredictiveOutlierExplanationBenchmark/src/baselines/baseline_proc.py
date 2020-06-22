from PredictiveOutlierExplanationBenchmark.src.baselines.micencova import CA_Lasso
import pandas as pd

from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset


def run_micencova(dataset):
    cal = CA_Lasso(dataset)
    cal.run()


if __name__ == '__main__':
    df = pd.read_csv('../results_normal/random_oversampling/lof/classification/datasets/synthetic/hics/group_g1/hics_100_g1_001/pseudo_samples_0/pseudo_samples_0_data.csv')
    dataset = Dataset(df, 'is_anomaly', 'subspaces')
    run_micencova(dataset)
