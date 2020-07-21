from Visualizer import Visualizer


if __name__ == '__main__':
    # hics_p = '../results/classification/datasets/synthetic/hics/group_g1'
    # Visualizer(hics_p, 'roc_auc').visualize(dims=100, original_data_viz=False)
    # path_real = '../results/classification/datasets/real/arrhythmia_015'
    path_real = '../results_predictive/iforest/protean/random_oversampling/classification/datasets/real/arrhythmia_015'
    Visualizer(path_real, 'roc_auc').visualize(dims=None, original_data_viz=False)
    pass