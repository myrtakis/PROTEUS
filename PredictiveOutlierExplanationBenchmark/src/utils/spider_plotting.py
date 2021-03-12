import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from sklearn import preprocessing


def construct_spider_plot(X, Y, anomaly_label, explanation_features, sample_ids_to_plot):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    medians = X.loc[:, explanation_features].median(axis=0)
    fig = go.Figure()

    q25 = X.loc[:, explanation_features].quantile(0.25)
    q75 = X.loc[:, explanation_features].quantile(0.75)

    iqr = q75-q25

    for pid in sample_ids_to_plot:
            name = 'Anomaly id' + str(pid) if Y[pid] == anomaly_label else 'Normal id' + str(pid)
            dists_from_median = np.absolute(medians.values - X.loc[pid, explanation_features].values)
            fig.add_trace(go.Scatterpolar(
                r=dists_from_median,
                theta=explanation_features,
                # fill='toself',
                name=name
            ))

    # fig.add_trace(go.Scatterpolar(
    #     r=iqr,
    #     theta=explanation_features,
    #     fill='toself',
    #     name='IQR',
    #     marker=None
    # ))
    fig = px.line_polar(r=iqr, theta=explanation_features, line_close=True)

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    # fig.show()
    fig.write_image("testspider.eps")


if __name__=='__main__':
    expl_size = '6'
    detector = 'iforest'
    dataset = 'arrhythmia_015'

    base_path = 'results_predictive_grouping/' + detector + '/protean/random_oversampling/classification/datasets/real/' +\
                        dataset + '/pseudo_samples_0/noise_0'
    results_path = base_path + '/expl_size_' + expl_size + '/best_model.json'

    data = base_path + '/pseudo_samples_0_data.csv'

    X = pd.read_csv(data)
    Y = X['is_anomaly'].values
    X = X.drop(columns=['is_anomaly'])

    anomaly_ids = np.where(Y == 1)[0]
    normal_ids = np.where(Y == 0)[0]

    with open(results_path) as f:
        results = json.load(f)
    expl_features = list(X.columns[results['fs']['roc_auc']['feature_selection']['features']])
    anomalies = [96, 133, 185]
    normals = [0,1, 122]
    samples_to_plot = [*anomalies]
    construct_spider_plot(X=X, Y=Y, anomaly_label=1, explanation_features=expl_features, sample_ids_to_plot=samples_to_plot)
