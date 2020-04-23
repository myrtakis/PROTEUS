import argparse
from PredictiveOutlierExplanationBenchmark.src.visualization.VizProcessor import VizProcessor


if __name__ == '__main__':
    hics_p = '../results/classification/datasets/real/arrhythmia/navigator.json'
    VizProcessor.process(hics_p)
    pass