from PredictiveOutlierExplanationBenchmark.src.analysis.visualization.Visualizer import Visualizer


if __name__ == '__main__':
    hics_p = '../results/classification/datasets/synthetic/hics/group_g1'
    path_real = '../results/classification/datasets/real/arrhythmia_005'
    viz_proc = Visualizer(path_real, 'roc_auc').visualize(dims=None, original_data_viz=False)
    pass