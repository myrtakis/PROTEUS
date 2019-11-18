from sklearn.neighbors import LocalOutlierFactor


def lof(dataframe, params):
    if 'knn' not in params:
        params['knn'] = 15  # default value
    clf = LocalOutlierFactor(n_neighbors=int(params['knn']), algorithm='brute', metric='euclidean', contamination='auto')
    clf.fit_predict(dataframe)
    scores = clf.negative_outlier_factor_
    scores = -scores
    return scores
