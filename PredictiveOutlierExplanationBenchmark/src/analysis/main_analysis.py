import argparse

# from DetectorsAnalysis import DetectorAnalysis
# from RelFeaturesRatioAnalysis import RelFeaturesRatio
from PerformanceAnalysis import PerfAnalysis
from  DetectorsAnalysis import DetectorAnalysis


if __name__ == '__main__':
    # path = '../results_predictive/iforest/protean/random_oversampling/classification/datasets/synthetic'
    path = '../results_predictive/loda/protean/random_oversampling/classification/datasets/real'
    # path = '../results_normal/random_oversampling/iforest/classification/datasets/synthetic/hics/group_g1'
    # path = '../results_predictive/random_oversampling/iforest/classification/datasets/synthetic/hics/group_g1'
    # path = '../results_normal/random_oversampling/lof/classification/datasets/real'
    metric_id = 'roc_auc'
    # RelFeaturesRatio(path, metric_id).analyze()

    # DetectorAnalysis(path, metric_id).analyze(real_data=False)
    DetectorAnalysis(path, metric_id).analyze(real_data=True)
    # DetectorAnalysis(path, metric_id).analyze_hold_out_effectiveness()

    # PerfAnalysis(path, metric_id, hold_out_effectiveness=False).analyze(original_data_analysis=False, real_data=True)
    # print('Hold out effectiveness')
    # PerfAnalysis(path, metric_id, hold_out_effectiveness=True).analyze(original_data_analysis=False, real_data=True)

    # PerfAnalysis(path, metric_id, hold_out_effectiveness=False).analyze(original_data_analysis=False, real_data=False)
    # PerfAnalysis(path, metric_id, hold_out_effectiveness=True).analyze(original_data_analysis=False, real_data=False)
