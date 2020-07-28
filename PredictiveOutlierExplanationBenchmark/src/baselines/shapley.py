from collections import OrderedDict
import pandas as pd
import shap
import numpy as np


class SHAP:

    def __init__(self, dataset_detected_outliers, detector_predict_func):
        self.dataset = dataset_detected_outliers
        assert detector_predict_func is not None
        self.predict_func = detector_predict_func

    def run(self):
        outlier_inds = self.dataset.get_outlier_indices()
        outliers = self.dataset.get_X().iloc[outlier_inds, :]
        if self.dataset.get_X().shape[1] > 5000:
            rand_inds = np.random.choice(len(outlier_inds), 5, replace=False)
            ref_set = outliers.iloc[rand_inds, :]
        else:
            ref_set = outliers
        det_shap_vals = shap.KernelExplainer(self.predict_func, ref_set)
        local_explanations = det_shap_vals.shap_values(outliers)
        return pd.DataFrame(local_explanations, index=outlier_inds, columns=np.arange(self.dataset.get_X().shape[1]))
