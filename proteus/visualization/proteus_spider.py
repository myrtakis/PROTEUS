import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

    
def transofrm_values_to_quantiles(pid, X):
    quant_vals = []
    for c in X.columns:
        sorted_vals = np.sort(X.loc[:, c].values)
        quantile_of_val = np.where(sorted_vals == X.loc[pid, c])[0][-1] / X.shape[0]
        quantile_of_val = 1 - quantile_of_val if quantile_of_val < 0.25 else quantile_of_val
        quant_vals.append(quantile_of_val)
    return quant_vals


def draw_iqr(ax, angles):
    q25 = np.full(len(angles)-1, 0.25)
    q75 = np.full(len(angles)-1, 0.75)

    ax.plot(angles, np.concatenate((q75, [q75[0]])), '-', linewidth=.5, color='tab:green')
    ax.fill(angles, np.concatenate((q75, [q75[0]])), alpha=0.1, color='tab:green')

    ax.plot(angles, np.concatenate((q25, [q25[0]])), '-', linewidth=.5, color='tab:green')
    ax.fill(angles, np.concatenate((q25, [q25[0]])), alpha=1, color='white')


def spider_plot(X, sample_ids_to_plot):
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, X.shape[1], endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for pid in sample_ids_to_plot:
        point_vals = transofrm_values_to_quantiles(pid, X)
        stats = np.concatenate((point_vals, [point_vals[0]]))
        ax.plot(angles, stats, label= pid+1)

    draw_iqr(ax, angles)

    ax.set_thetagrids(angles * 180 / np.pi, list(X.columns))
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    plt.legend(loc="best")
    plt.show()