from pathlib import Path
import json
from sklearn import manifold
from sklearn.decomposition import PCA
from PredictiveOutlierExplanationBenchmark.src.utils import utils
from PredictiveOutlierExplanationBenchmark.src.utils.pseudo_samples import PseudoSamplesMger
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import *
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from adjustText import adjust_text


class VizProcessor:

    __save_dir = 'figures/vizualizations'

    __dim_reduction_methods = {
        'pca':
            {
                'method': PCA(n_components=2, random_state=0),
                'axis_labels': ['Principal Component 1', 'Principal Component 2']
             },
        'tsne':
            {
                'method': manifold.TSNE(n_components=2, init='pca', perplexity=40, random_state=0),
                'axis_labels': ['t-SNE Embedding 1', 't-SNE Embedding 2']
             }
    }

    def __init__(self, path_to_dir, metric_id):
        self.original_dataset = None
        self.is_anomaly_column_original = None
        self.path_to_dir = path_to_dir
        self.nav_files = None
        self.nav_file = None
        self.metric_id = metric_id
        self.read_original_dataset()

    def read_original_dataset(self):
        self.original_dataset = pd.read_csv(self.nav_file[FileKeys.navigator_original_dataset_path])
        if 'subspaces' in self.original_dataset.columns:
            self.original_dataset = self.original_dataset.drop(columns=['subspaces'])
        self.is_anomaly_column_original = self.original_dataset['is_anomaly']
        self.original_dataset = self.original_dataset.drop(columns=['is_anomaly'])

    def __read_nav_files(self):
        nav_files = []
        nav_files_paths = utils.get_files_recursively(self.path_to_dir, FileNames.navigator_fname)
        for f in nav_files_paths:
            with open(f) as json_file:
                nav_files.append(json.load(json_file))
        return nav_files

    def __sort_files_by_dim(self):
        nav_files_sort_by_dim = {}
        for nfile in self.nav_files:
            data_dim = pd.read_csv(nfile[FileKeys.navigator_original_dataset_path]).shape[1]
            nav_files_sort_by_dim[data_dim] = nfile
        return dict(sorted(nav_files_sort_by_dim.items()))

    def visualize(self, dims=None, original_data_viz=False):
        self.nav_files = self.__read_nav_files()
        self.nav_files = self.__sort_files_by_dim()
        self.__visualize_nav_files(dims)

    def __visualize_nav_files(self, dims):
        for dim, nav_file in self.nav_files.items():
            if dims is not None and dim != dims:
                continue
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=True)


    @staticmethod
    def process(path_to_navigator_file):
        with open(path_to_navigator_file) as json_file:
            VizProcessor.__nav_file = json.load(json_file)
        VizProcessor.__load_original_dataset()
        ps_samples = VizProcessor.__nav_file[FileKeys.navigator_pseudo_samples_key]
        best_benchmark_results_per_metric = utils.optimal_pseudo_samples_per_metric(ps_samples, False)
        best_explanations = VizProcessor.__best_explanation_per_metric(best_benchmark_results_per_metric)
        for m_id, best_expl in best_explanations.items():
            print(m_id)
            print('best model', best_benchmark_results_per_metric[m_id]['best_model'])
            key = next(iter(best_expl))
            print('best explanation alg', key, best_expl[key]['effectiveness'])
            print('Best K', best_benchmark_results_per_metric[m_id]['k'])
            df, path_to_best_model_dir = VizProcessor.__load_dataset_of_best_model(best_benchmark_results_per_metric[m_id]['k'])
            parent_dir = VizProcessor.__get_create_parent_output_dir(path_to_best_model_dir, m_id)
            VizProcessor.__viz_coordinator(parent_dir, df, best_expl)
            print('---------')
            break

    def __viz_coordinator(self, parent_dir, df, best_explanation):
        key = next(iter(best_explanation))
        features = best_explanation[key]['feature_selection']['features']
        # todo delete following line
        features = features[0:10] if len(features) > 10 else features
        if len(features) == 2:
            self.__viz_in_2d(df, features, None, parent_dir, 'viz2d.png')
        elif len(features) == 3:
            print('Visualization in 3d not ready yet!!!')
            pass
        elif len(features) > 3:
            for method, data in VizProcessor.__dim_reduction_methods.items():
                VizProcessor.__viz_in_2d(df, data['axis_labels'], data['method'], parent_dir, method+'.png')
        else:
            assert False, len(features)

    def __viz_in_2d(self, df, axis_labels, transform_func, parent_dir, fname):
        pseudo_samples_indices = VizProcessor.__get_artificial_point_indices(df)
        zindex = VizProcessor.__zindex_of_points(df.drop(df.index[pseudo_samples_indices]))
        colors = VizProcessor.__colors_of_points(df.drop(df.index[pseudo_samples_indices]))
        texts = VizProcessor.__texts_of_points(df.drop(df.index[pseudo_samples_indices]))
        cols_to_drop = ['is_anomaly']
        if 'subspaces' in df.columns:
            cols_to_drop.append('subspaces')
        new_df = df.drop(columns=cols_to_drop)
        if transform_func is not None:
            new_df = transform_func.fit_transform(new_df)
            new_df = np.delete(new_df, pseudo_samples_indices, axis=0)
        if not isinstance(new_df, np.ndarray):
            new_df = new_df.values()
        plt.scatter(new_df[zindex, 0], new_df[zindex, 1], c=colors[zindex], cmap='Spectral')
        annotations = []
        for i in zindex:
            if texts[i] != '':
                annotations.append(plt.text(new_df[i, 0], new_df[i, 1], texts[i], color='brown', fontweight='bold', size=9))
        adjust_text(annotations, autoalign='')
        plt.title("%s" % (fname.replace('.png', '')))
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        # plt.savefig(Path(parent_dir, fname), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.clf()

    def __colors_of_points(self, df):
        outliers = VizProcessor.__get_outlier_indices(df)
        colors = np.full(df.shape[0], 'skyblue', dtype=object)
        colors[outliers] = 'r'
        return colors

    def __texts_of_points(self, df):
        inliers = VizProcessor.__get_inlier_indices(df)
        text = np.array([str(x) for x in range(df.shape[0])], dtype=object)
        text[inliers] = ''
        return text

    def __zindex_of_points(self, df):
        inliers = VizProcessor.__get_inlier_indices(df)
        outliers = VizProcessor.__get_outlier_indices(df)
        return np.array([*inliers, *outliers])

    @staticmethod
    def __get_inlier_indices(df):
        return np.array(df.loc[df['is_anomaly'] == 0, 'is_anomaly'].index)

    @staticmethod
    def __get_outlier_indices(df):
        return np.array(df.loc[df['is_anomaly'] == 1, 'is_anomaly'].index)

    def __get_artificial_point_indices(self, df):
        if self.original_dataset.shape[0] == df.shape[0]:
            return None
        else:
            return np.arange(self.original_dataset.shape[0], df.shape[0])

    @staticmethod
    def __get_create_parent_output_dir(path_to_best_model_dir, metric):
        parent_dir = Path(path_to_best_model_dir, VizProcessor.__save_dir, metric)
        parent_dir.mkdir(parents=True, exist_ok=True)
        return parent_dir
