import pandas as pd
from sklearn.ensemble import IsolationForest

rel_feature_subsets_col_name = 'subspaces'
outlier_col_name = 'is_anomaly'
dataset_path = '../datasets/breast_lof_031_010.csv'


def run_iforest_fit_predict(dataset_path):
    df = pd.read_csv(dataset_path)
    #samples = df.drop(columns=)
    clf = IsolationForest(n_estimators=100, max_samples=256, behaviour='new', contamination='auto')
    train_set = df[1:100]
    print(train_set.index)
    #clf.fit()


if __name__ == '__main__':
    run_iforest_fit_predict(dataset_path)
