from matplotlib.font_manager import FontProperties
from PredictiveOutlierExplanationBenchmark.src.configpkg import ConfigMger, DatasetConfig
from PredictiveOutlierExplanationBenchmark.src.holders.Dataset import Dataset
from PredictiveOutlierExplanationBenchmark.src.models.OutlierDetector import Detector
from PredictiveOutlierExplanationBenchmark.src.utils.helper_functions import read_nav_files, sort_files_by_dim
from PredictiveOutlierExplanationBenchmark.src.utils.pseudo_samples import PseudoSamplesMger
from PredictiveOutlierExplanationBenchmark.src.utils.shared_names import FileKeys
from sklearn.metrics import roc_auc_score
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


conf = {'path': '../predictive/lof', 'detector': 'lof', 'type': 'synthetic'}
# conf = {'path': '../predictive/iforest', 'detector': 'iforest', 'type': 'synthetic'}
# conf = {'path': '../predictive/lof', 'detector': 'lof', 'type': 'real'}


if __name__ == '__main__':
    pass