import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


rel_feature_subsets_col_name = 'subspaces'
outlier_col_name = 'is_anomaly'
dataset_path = '../datasets/breast_lof_031_010.csv'


def iforest_train(X_train):
    return IsolationForest(n_estimators=100, max_samples=256, behaviour='new', contamination='auto').fit(X_train)


def iforest_predict(model, X_test):
    return np.array(model.score_samples(X_test)) * -1


def lof_train(X):
    return LocalOutlierFactor(n_neighbors=15, novelty=True, contamination='auto').fit(X)


def lof_predict(model, X_test):
    return np.array(model.score_samples(X_test)) * -1


def add_gaussian_noise(df, detector_id):
    mu, sigma = 0, 0.1
    replicas_per_outlier = 5
    outlier_inds = list(df[df[outlier_col_name] == 1].index)
    inlier_inds = list(df[df[outlier_col_name] == 0].index)
    X = df.drop(columns=outlier_col_name)
    model = detectors_map[detector_id]['train'](X)
    points_scores = detectors_map[detector_id]['predict'](model, X)
    for ind in outlier_inds:
        outlier_values = X.iloc[ind, :].values
        for r in range(0, replicas_per_outlier):
            noise = np.random.normal(mu, sigma, [1, X.shape[1]])
            noisy_point = outlier_values + noise
            series = pd.Series(*noisy_point, index=list(X.columns))
            test_sample = pd.DataFrame([series])
            new_point_score = detectors_map[detector_id]['predict'](model, test_sample)
            label = get_new_point_label(inlier_inds, points_scores, new_point_score)
            noisy_point = np.append(noisy_point, label)
            df = df.append(pd.Series(noisy_point, index=df.columns), ignore_index=True)
    return df


def get_new_point_label(inlier_inds, points_scores, new_point_score):
    max_inlier_score = max(points_scores[inlier_inds])
    return int(new_point_score[0] >= max_inlier_score)


detectors_map = {
    'iforest': {'train': iforest_train, 'predict': iforest_predict},
    'lof': {'train': lof_train, 'predict': lof_predict}
}


if __name__ == '__main__':
    df = pd.read_csv(dataset_path)
    df = df.drop(columns=rel_feature_subsets_col_name)
    new_df = add_gaussian_noise(df, 'lof')
    new_df.to_csv('../datasets/breast_lof_noise_031_010.csv', index=False)
