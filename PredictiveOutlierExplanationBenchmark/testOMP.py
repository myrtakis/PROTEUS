from sklearn.linear_model import OrthogonalMatchingPursuit
import pandas as pd
import numpy as np


def run_omp(df):
    subspaces = set(df[df['subspaces'] != '-']['subspaces'])
    y = df['is_anomaly']
    X = df.drop(columns=['is_anomaly', 'subspaces'])
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10).fit(X, y)
    print(np.where(omp.coef_ != 0)[0])


if __name__=='__main__':
    import warnings
    warnings.warn('FBED is removed from this version of PROTEUS to avoid conflicts with Apache 2.0 license. FBED returns an empty array of features.')
    print('Relevant Features:', [0,1,2,3,4])
    for i in np.arange(20, 120, 20):
        print('Dimensions', i)
        path = 'datasets/synthetic/hics/group_g1/hics_' + str(i) + '_g1.csv'
        run_omp(pd.read_csv(path))