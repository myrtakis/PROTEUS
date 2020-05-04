import argparse

from PredictiveOutlierExplanationBenchmark.src.analysis.DetectorsAnalysis import DetectorAnalysis
from PredictiveOutlierExplanationBenchmark.src.analysis.RelFeaturesRatioAnalysis import RelFeaturesRatio
from PredictiveOutlierExplanationBenchmark.src.analysis.PerformanceAnalysis import PerfAnalysis


if __name__ == '__main__':
    path = '../results/classification/datasets/synthetic/hics/group_g1'
    metric_id = 'roc_auc'
    RelFeaturesRatio(path, metric_id).analyze()

    # DetectorAnalysis(path, metric_id).analyze()

    # PerfAnalysis(path, metric_id).analyze(original_data_analysis=True)
