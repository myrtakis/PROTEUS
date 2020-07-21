from pathlib import Path
import json
from sklearn import manifold
from sklearn.decomposition import PCA
import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandpadir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandpadir)
from utils import helper_functions
from utils.pseudo_samples import PseudoSamplesMger
from utils.shared_names import *
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from adjustText import adjust_text


class Visualizer:

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
        self.path_to_dir = path_to_dir
        self.nav_files = None
        self.metric_id = metric_id

    def visualize(self, dims=None, original_data_viz=False):
        self.nav_files = self.__read_nav_files()
        self.nav_files = self.__sort_files_by_dim()
        self.__visualize_nav_files(dims)
        if original_data_viz:
            self.visualize_original_file(dims)

    def visualize_original_file(self, dims=None):
        for dim, nav_file in self.nav_files.items():
            if dims is not None and dim < dims:
                continue
            self.read_original_dataset(nav_file)
            original_data = nav_file[FileKeys.navigator_original_data]
            best_model = helper_functions.get_best_model_original_data(original_data, self.metric_id, True)
            self.__viz_coordinator(parent_dir=None, df=self.original_dataset, features=best_model['feature_selection']['features'], keep_only_given_features=True)
            self.__viz_coordinator(parent_dir=None, df=self.original_dataset, features=np.arange(0, self.original_dataset.shape[1] - 2),
                                   keep_only_given_features=False)

    def __visualize_nav_files(self, dims):
        for dim, nav_file in self.nav_files.items():
            if dims is not None and dim < dims:
                continue
            self.read_original_dataset(nav_file)
            ps_mger = PseudoSamplesMger(nav_file[FileKeys.navigator_pseudo_samples_key], self.metric_id, fs=True)
            best_model, k = ps_mger.get_best_model()
            print('Best k', k)
            dataset_path_of_best_k = ps_mger.get_info_field_of_k(k, FileKeys.navigator_pseudo_samples_data_path)
            df = pd.read_csv(dataset_path_of_best_k)
            print('Explained Boundary')
            self.__viz_coordinator(parent_dir=None, df=df, features=best_model['feature_selection']['features'], keep_only_given_features=True)
            print('Original Boundary')
            self.__viz_coordinator(parent_dir=None, df=df, features=np.arange(0, self.original_dataset.shape[1] - 2), keep_only_given_features=False)

    def read_original_dataset(self, nav_file):
        self.original_dataset = pd.read_csv(nav_file[FileKeys.navigator_original_dataset_path])

    def __read_nav_files(self):
        nav_files = []
        nav_files_paths = helper_functions.get_files_recursively(self.path_to_dir, FileNames.navigator_fname)
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

    def __viz_coordinator(self, parent_dir, df, features, keep_only_given_features=True):
        # todo delete following line
        if keep_only_given_features:
            features = features[0:10] if len(features) > 10 else features
        if len(features) == 2:
            self.__viz_in_2d(df,features, features, None, parent_dir, 'viz2d.png')
        elif len(features) == 3:
            print('Visualization in 3d not ready yet!!!')
            pass
        elif len(features) > 3:
            for method, data in Visualizer.__dim_reduction_methods.items():
                self.__viz_in_2d(df, features, data['axis_labels'], data['method'], parent_dir, method + '.png')
        else:
            assert False, len(features)

    def __viz_in_2d(self, df, features, axis_labels, transform_func, parent_dir, fname):
        pseudo_samples_indices = self.__get_artificial_point_indices(df)
        reduced_df = df.copy()
        if pseudo_samples_indices is not None and len(pseudo_samples_indices) > 0:
            reduced_df = df.drop(df.index[pseudo_samples_indices])
        zindex = self.__zindex_of_points(reduced_df)
        colors = self.__colors_of_points(reduced_df)
        texts = self.__texts_of_points(reduced_df)
        cols_to_drop = ['is_anomaly']
        if 'subspaces' in df.columns:
            cols_to_drop.append('subspaces')
        new_df = df.iloc[:, features]
        if transform_func is not None:
            new_df = transform_func.fit_transform(new_df)
            if pseudo_samples_indices is not None and len(pseudo_samples_indices) > 0:
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
        outliers = Visualizer.__get_outlier_indices(df)
        colors = np.full(df.shape[0], 'skyblue', dtype=object)
        fp = self.__get_false_positive_indices(outliers)
        tp = self.__get_true_outliers_indices(outliers)
        high_scored_points = np.concatenate((fp, tp)).astype(int)
        colors[high_scored_points] = 'r'
        # colors[tp] = 'r'
        # colors[fp] = 'gold'
        return colors

    def __texts_of_points(self, df):
        inliers = Visualizer.__get_inlier_indices(df)
        text = np.array([str(x) for x in range(df.shape[0])], dtype=object)
        text[inliers] = ''
        return text

    def __zindex_of_points(self, df):
        inliers = Visualizer.__get_inlier_indices(df)
        outliers = Visualizer.__get_outlier_indices(df)
        return np.array([*inliers, *outliers])

    def __get_false_positive_indices(self, detected_outliers):
        outlier_inds_original = Visualizer.__get_outlier_indices(self.original_dataset)
        return list(set(detected_outliers) - set(outlier_inds_original))

    def __get_true_outliers_indices(self, detected_outliers):
        outlier_inds_original = Visualizer.__get_outlier_indices(self.original_dataset)
        return list(set(detected_outliers).intersection(set(outlier_inds_original)))

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
        parent_dir = Path(path_to_best_model_dir, Visualizer.__save_dir, metric)
        parent_dir.mkdir(parents=True, exist_ok=True)
        return parent_dir
