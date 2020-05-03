from pathlib import Path
import json
from sklearn import manifold
from sklearn.decomposition import PCA
from PredictiveOutlierExplanationBenchmark.src.utils import utils
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

    def __init__(self, path_to_navigator_file):
        self.original_dataset = None
        self.nav_file = None
        self.read_nav_file(path_to_navigator_file)
        self.read_original_dataset()

    def read_original_dataset(self):
        self.original_dataset = pd.read_csv(self.nav_file[FileKeys.navigator_original_dataset_path])

    def read_nav_file(self, path_to_navigator_file):
        with open(path_to_navigator_file) as json_file:
            self.nav_file = json.load(json_file)


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

    @staticmethod
    def __viz_coordinator(parent_dir, df, best_explanation):
        key = next(iter(best_explanation))
        features = best_explanation[key]['feature_selection']['features']
        print(features)
        opt_features = set(np.arange(0,5).tolist())
        print('fprecision', len(opt_features.intersection(features)) / len(features))
        print('frecall', len(opt_features.intersection(features)) / len(opt_features))
        if len(features) == 2:
            VizProcessor.__viz_in_2d(df, features, None, parent_dir, 'viz2d.png')
        elif len(features) == 3:
            pass
        elif len(features) > 3:
            for method, data in VizProcessor.__dim_reduction_methods.items():
                VizProcessor.__viz_in_2d(df, data['axis_labels'], data['method'], parent_dir, method+'.png')
        else:
            assert False, len(features)


    @staticmethod
    def __viz_in_2d(df, axis_labels, transform_func, parent_dir, fname):
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
        plt.savefig(Path(parent_dir, fname), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.clf()


    @staticmethod
    def __load_original_dataset():
        VizProcessor.__original_dataset = pd.read_csv(VizProcessor.__nav_file[FileKeys.navigator_original_dataset_path])

    @staticmethod
    def __load_dataset_of_best_model(best_k):
        for ps_key, ps_data in VizProcessor.__nav_file[FileKeys.navigator_pseudo_samples_key].items():
            if ps_data[FileKeys.navigator_pseudo_samples_num_key] == best_k:
                path_to_best_model_dir = ps_data[FileKeys.navigator_pseudo_sample_dir_key]
                df = pd.read_csv(VizProcessor.__get_dataset_in_dir(path_to_best_model_dir))
                if 'subspaces' in df.columns:
                    df.drop(columns=['subspaces'])
                return df, path_to_best_model_dir

    @staticmethod
    def __colors_of_points(df):
        outliers = VizProcessor.__get_outlier_indices(df)
        colors = np.full(df.shape[0], 'skyblue', dtype=object)
        colors[outliers] = 'r'
        return colors

    @staticmethod
    def __texts_of_points(df):
        inliers = VizProcessor.__get_inlier_indices(df)
        text = np.array([str(x) for x in range(df.shape[0])], dtype=object)
        text[inliers] = ''
        return text

    @staticmethod
    def __zindex_of_points(df):
        inliers = VizProcessor.__get_inlier_indices(df)
        outliers = VizProcessor.__get_outlier_indices(df)
        return np.array([*inliers, *outliers])

    @staticmethod
    def __get_inlier_indices(df):
        return np.array(df.loc[df['is_anomaly'] == 0, 'is_anomaly'].index)

    @staticmethod
    def __get_outlier_indices(df):
        return np.array(df.loc[df['is_anomaly'] == 1, 'is_anomaly'].index)

    @staticmethod
    def __get_artificial_point_indices(df):
        if VizProcessor.__original_dataset.shape[0] == df.shape[0]:
            return None
        else:
            return np.arange(VizProcessor.__original_dataset.shape[0], df.shape[0])

    @staticmethod
    def __best_explanation_per_metric(best_benchmark_results_per_metric):
        best_explanations = {}
        for m_id, data in best_benchmark_results_per_metric.items():
            best_expl = utils.best_model_dict(data['bench_results'], fs=True)
            best_explanations.update(utils.best_conf_in_repetitions(data['bench_results'], best_expl[m_id]['conf_id'], m_id))
        return best_explanations

    @staticmethod
    def __get_dataset_in_dir(path_to_dir):
        for f in os.listdir(path_to_dir):
            if f.endswith('.csv'):
                return Path(path_to_dir, f)

    @staticmethod
    def __get_create_parent_output_dir(path_to_best_model_dir, metric):
        parent_dir = Path(path_to_best_model_dir, VizProcessor.__save_dir, metric)
        parent_dir.mkdir(parents=True, exist_ok=True)
        return parent_dir
