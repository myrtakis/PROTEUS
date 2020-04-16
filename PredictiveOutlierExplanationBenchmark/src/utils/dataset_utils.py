from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arff2pandas import a2p


anomaly_column = 'is_anomaly'


def convert_arff_to_csv(path, split_cols_to_char=None):
    with open(path) as f:
        df = a2p.load(f)
    if split_cols_to_char is not None:
        cols = [col.split(split_cols_to_char)[0] for col in df.columns]
        cols[-1] = anomaly_column
        df = df.set_axis(cols, axis=1, inplace=False)
    df = preprocess(df, anomaly_column)
    return df


def convert_mat_to_csv(path):
    mat = loadmat(path)
    vars = np.arange(mat['X'].shape[1]+1)
    vars = ['Var' + str(v) for v in vars]
    vars[-1] = anomaly_column
    csv = np.hstack((mat['X'], mat['y']))
    path = path.replace('mat', 'csv')
    df = pd.DataFrame(csv, columns=vars)
    df = preprocess(df, anomaly_column)
    df.to_csv(path, header=False, index=False)
    return df


def preprocess(df, target_column, subspace_column=None):
    df = remove_nan_columns(df, target_column)
    df = remove_features_of_single_value(df, target_column)
    df = remove_duplicates(df)
    df = df.apply(pd.to_numeric)
    df = min_max_normalization(df, target_column, subspace_column)
    return df


def min_max_normalization(df, target_col_name, subspace_col_name=None):
    scaler = MinMaxScaler()
    X = df.drop(columns=[target_col_name])
    if subspace_col_name is not None:
        X = df.drop(columns=[subspace_col_name])
        ground_truth = df.loc[:, [target_col_name, subspace_col_name]]
    else:
        ground_truth = df.loc[:, [target_col_name]]
    X_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return pd.concat([X, ground_truth], axis=1)


def remove_nan_columns(df, target_column):
    y = df[target_column]
    df = df.drop(columns=[target_column])
    nan_cols = df.columns[df.isna().any()].tolist()
    if len(nan_cols) > 0:
        df = df.drop(columns=nan_cols)
    return pd.concat([df, y], axis=1)


def remove_features_of_single_value(df, target_column):
    y = df[target_column]
    df = df.drop(columns=[target_column])
    unq = df.nunique()
    one_val_columns = list(unq[unq == 1].index)
    if len(one_val_columns) > 0:
        df = df.drop(columns=one_val_columns)
    return pd.concat([df, y], axis=1)


def remove_duplicates(df):
    return df