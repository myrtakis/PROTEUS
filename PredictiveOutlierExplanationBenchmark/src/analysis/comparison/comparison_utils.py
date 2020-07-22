import os
from pathlib import Path

import numpy as np
import json

from utils.shared_names import FileNames


def load_baseline_explanations(baseline_path, max_features=None):
    baseline_explanations_dict = {}
    explanation_file_path = Path(baseline_path, FileNames.baselines_explanations_fname)
    with open(explanation_file_path) as json_file:
        for method, data in json.load(json_file).items():
            global_explanation_sorted = np.argsort(data['global_explanation'])[::-1]
            max_features = len(global_explanation_sorted) if max_features is None else max_features
            baseline_explanations_dict[method] = list(global_explanation_sorted[0:max_features])
    return baseline_explanations_dict


def get_dataset_name(dataset_path, synthetic):
    if synthetic:
        return 'S'
    else:
        return os.path.splitext(os.path.basename(dataset_path))[0]