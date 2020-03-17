import os
from collections import OrderedDict
from functools import partial
from time import time, strftime, gmtime

from adjustText import adjust_text
from multipledispatch import dispatch
from matplotlib import rc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
import umap
from sklearn import manifold
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap



def pca_visualization(df, savedir):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df.drop(columns=['is_anomaly']))
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['pc1', 'pc2'])
    finalDf = pd.concat([principalDf, df[['is_anomaly']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['b', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['is_anomaly'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
                   , finalDf.loc[indicesToKeep, 'pc2']
                   , c=color
                   )
    ax.legend(targets)
    ax.grid()
    plt.show()
    plt.clf()


def ica_visualization(df, savedir):
    ica = FastICA(n_components=2, random_state=0)
    ica_trans = ica.fit_transform(df.drop(columns=['is_anomaly']))
    principalDf = pd.DataFrame(data=ica_trans
                               , columns=['ica1', 'ica2'])
    finalDf = pd.concat([principalDf, df[['is_anomaly']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('ICA 1', fontsize=15)
    ax.set_ylabel('ICA 2', fontsize=15)
    # ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['b', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['is_anomaly'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'ica1']
                   , finalDf.loc[indicesToKeep, 'ica2']
                   , c=color
                   )
    ax.legend(targets)
    ax.grid()
    plt.show()
    plt.clf()


def umap_visualization(df, savedir):
    um = umap.UMAP(n_neighbors=25, random_state=0, n_components=2)
    X_train = df.drop(columns=['is_anomaly'])
    Y_train = df['is_anomaly']
    trans = um.fit(X_train)
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=25, c=Y_train, cmap='Spectral')
    plt.show()


def tsne_visualization(df, savedir):
    final_df = df.iloc[:, 0:df.shape[1] - 2]
    feat_cols = list(range(final_df.shape[1]))
    final_df['y'] = df.iloc[:, df.shape[1] - 1]
    final_df['label'] = final_df['y'].apply(lambda i: str(i))
    tsne_results = manifold.TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(df.iloc[:, feat_cols])

    final_df['tsne-axis-one'] = tsne_results[:, 0]
    final_df['tsne-axis-two'] = tsne_results[:, 1]

    colors = ["#383838", "#FF0B04"]

    sns.scatterplot(
        x="tsne-axis-one", y="tsne-axis-two",
        hue="y",
        palette=sns.color_palette(colors),
        data=final_df,
        legend="full",
    )

    # dataset_name = os.path.splitext(os.path.basename(f))[0]
    # output = os.path.join(savedir, 'tsne_' + dataset_name + '.png')
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)

    # plt.title(dataset_name)
    # plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()


def visualize_selected_features(df, feature_ids, savedir):
    if len(feature_ids) >= 3:
        dim_reduction_vizualizations(df, feature_ids, savedir)
        actual_features_vizualizations(df, feature_ids, savedir)
    else:
        actual_features_vizualizations(df, feature_ids, savedir)


def actual_features_vizualizations(df, feature_ids, savedir):
    inliers = np.array(df.loc[df['is_anomaly'] == 0, 'is_anomaly'].index).tolist()
    outliers = np.array(df.loc[df['is_anomaly'] == 1, 'is_anomaly'].index).tolist()
    zindex_points = [*inliers, *outliers]
    X = df.drop(columns=['is_anomaly'])
    # Fixing z-index to be higher for outliers (they will be rendered last)
    X = pd.concat([X.iloc[inliers], X.iloc[outliers]], axis=0)
    fig = plt.figure()
    colors = get_colors(len(inliers), len(outliers))
    if len(feature_ids) == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, cmap='Spectral')
        ax.set_title('Visualization 2D')
        ax.set_xlabel('F' + str(feature_ids[0]))
        ax.set_ylabel('F' + str(feature_ids[1]))
        texts = []
        for j, p in enumerate(zindex_points):
            if p in outliers:
                texts.append(ax.text(X.iloc[j, 0], X.iloc[j, 1], p,
                                     color='brown',
                                     fontweight='bold',
                                     # bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 1},
                                     size=9))
        adjust_text(texts, autoalign='')  # , arrowprops=dict(arrowstyle="->", color='black', lw=0.5, shrinkA=5))

    if len(feature_ids) == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=colors, cmap='Spectral')
        ax.set_title('Visualization 3D')
        ax.set_xlabel('F' + str(feature_ids[0]))
        ax.set_ylabel('F' + str(feature_ids[1]))
        ax.set_zlabel('F' + str(feature_ids[2]))
        for j, p in enumerate(zindex_points):
            if p in outliers:
                ax.text(X.iloc[j, 0], X.iloc[j, 1], X.iloc[j, 2], p,
                        color='brown',
                        fontweight='bold',
                        # bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 1},
                        size=9)
    # plt.tight_layout()
    # plt.show()
    dt_string = strftime("%d%m%Y%H%M%S", gmtime())
    final_output = os.path.join(savedir, 'actual_viz_' + dt_string + '.png')
    plt.savefig(final_output, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


def dim_reduction_vizualizations(df, feature_ids, savedir):
    inliers = np.array(df.loc[df['is_anomaly'] == 0, 'is_anomaly'].index).tolist()
    outliers = np.array(df.loc[df['is_anomaly'] == 1, 'is_anomaly'].index).tolist()
    zindex_points = [*inliers, *outliers]
    X = df.drop(columns=['is_anomaly'])
    # Fixing z-index to be higher for outliers (they will be rendered last)
    X = pd.concat([X.iloc[inliers], X.iloc[outliers]], axis=0)

    n_components = 2
    methods = {'PCA':
                   {'method': PCA(n_components=n_components, random_state=0),
                    'x_axis_name': 'Principal Component 1',
                    'y_axis_name': 'Principal Component 2'
                    },
               't-SNE':
                   {'method': manifold.TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0),
                    'x_axis_name': 't-SNE Embedding 1',
                    'y_axis_name': 't-SNE Embedding 2'
                    }
               }

    plt.figure(figsize=(13, 5.5))
    plt.subplots_adjust(wspace=0.2)
    plt.suptitle('Features = ' + str(feature_ids), fontsize=12, wrap=True)

    colors = get_colors(len(inliers), len(outliers))

    for i, (label, method) in enumerate(methods.items()):
        # if len(feature_ids) == 2:
        #     break
        plt.subplot(1, 2, i + 1)
        t0 = time()
        X_tr = method['method'].fit_transform(X)
        t1 = time()
        plt.scatter(X_tr[:, 0], X_tr[:, 1], c=colors, cmap='Spectral')
        plt.title("%s (%.2g sec)" % (label, t1 - t0))
        plt.xlabel(method['x_axis_name'])
        plt.ylabel(method['y_axis_name'])
        texts = []
        for j, p in enumerate(zindex_points):
            if p in outliers:
                texts.append(plt.text(X_tr[j, 0], X_tr[j, 1], p,
                                      color='brown',
                                      fontweight='bold',
                                      # bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 1},
                                      size=9))
        adjust_text(texts, autoalign='')  # , arrowprops=dict(arrowstyle="->", color='black', lw=0.5, shrinkA=5))

    # plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    # plt.show()
    dt_string = strftime("%d%m%Y%H%M%S", gmtime())
    final_output = os.path.join(savedir, 'dim_reduction_' + dt_string + '.png')
    plt.savefig(final_output, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


def manifold_visualizations(df, savedir):
    color = get_colors(df)

    X = df.drop(columns=['is_anomaly'])
    n_neighbors = 10
    n_components = 2
    fig = plt.figure(figsize=(15, 8))
    # fig.suptitle("Manifold Learning with %i points, %i neighbors"
    #              % (df.shape[0], n_neighbors), fontsize=14)
    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors, n_components, eigen_solver='auto')

    methods = OrderedDict()
    methods['LLE'] = LLE(method='standard')
    # methods['LTSA'] = LLE(method='ltsa')
    # methods['Hessian LLE'] = LLE(method='hessian')
    # methods['Modified LLE'] = LLE(method='modified')
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                     random_state=0)
    methods['UMAP'] = umap.UMAP(n_neighbors=n_neighbors, random_state=0, n_components=n_components)

    # Plot results
    for i, (label, method) in enumerate(methods.items()):
        t0 = time()
        X_tr = method.fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (label, t1 - t0))
        ax = fig.add_subplot(2, 3, i+1)
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=color, cmap='Spectral')
        ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        # ax.axis('tight')
        # ax.set(aspect='equal')

    plt.show()


@dispatch(pd.DataFrame)
def get_colors(df):
    colors = list(df['is_anomaly'].values)
    for i in range(0, len(colors)):
        colors[i] = 'r' if colors[i] == 1 else 'b'
    return colors


@dispatch(int, int)
def get_colors(num_inliers, num_outliers):
    colors = []
    for i in range(num_inliers + num_outliers):
        if i < num_inliers:
            colors.append('skyblue')
        else:
            colors.append('r')
    return colors


def visualize_2d(df, savedir):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    n = list(range(df.shape[0]))

    ind = df.loc[df['is_anomaly'] == 1, 'is_anomaly'].index
    text_ind = ind

    colors = np.array(['dimgray'] * df.shape[0], dtype=object)

    colors[ind] = 'red'

    fig, ax = plt.subplots()
    for i in range(len(x)):
        z_order = 2 if i in ind else 0
        color = colors[i]
        ax.scatter(x[i], y[i], c=color, zorder=z_order)

    for i, label in enumerate(n):
        if i in text_ind:
            ax.annotate(label, xy=(x[i], y[i]), c='dodgerblue', zorder=2)
    # plt.title(os.path.splitext(os.path.basename(dataset_path))[0])
    plt.xlabel('Feature ' + str(df.columns[0]))
    plt.ylabel('Feature ' + str(df.columns[1]))
    plt.show()
    # plt.savefig('visualizations/breast' + str(dims[0]) + str(dims[1]) + '.png', dpi=300)
