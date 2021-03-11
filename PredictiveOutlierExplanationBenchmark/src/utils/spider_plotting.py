import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np


def construct_spider_plot(X, Y, anomaly_label, explanation_features, sample_ids_to_plot):
    # categories = ['processing cost', 'mechanical properties', 'chemical stability',
    #               'thermal stability', 'device integration']

    # fig = go.Figure()

    # fig.add_trace(go.Scatterpolar(
    #     r=[1, 5, 2, 2, 3],
    #     theta=categories,
    #     fill='toself',
    #     name='Product A'
    # ))
    # fig.add_trace(go.Scatterpolar(
    #     r=[4, 3, 2.5, 1, 2],
    #     theta=categories,
    #     fill='toself',
    #     name='Product B'
    # ))
    #
    # fig.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             visible=True,
    #         )),
    # )

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=X.loc[sample_ids_to_plot['anomalies'], explanation_features].values[0],
        theta=explanation_features,
        #fill='toself',
        name='anomaly'
    ))

    fig.add_trace(go.Scatterpolar(
        r=X.loc[sample_ids_to_plot['normal'], explanation_features].values[0],
        theta=explanation_features,
        # fill='toself',
        name='normal'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))

    # fig.show()
    fig.write_image("testspider.png")



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

    construct_spider_plot(X=X, Y=Y, anomaly_label=1, explanation_features=expl_features, sample_ids_to_plot={'anomalies': [anomaly_ids[0]],
                                                                                                             'normal': [normal_ids[0]]})
    pass