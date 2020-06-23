import shap
import numpy as np


class SHAP:

    def __init__(self, dataset_detected_outliers, detector_predict_func):
        self.dataset = dataset_detected_outliers
        assert detector_predict_func is not None
        self.predict_func = detector_predict_func

    def run(self):
        local_explanations = {}
        for o in self.dataset.get_outlier_indices():
            outlier_point = self.dataset.get_X().iloc[o, :]
            det_shap_vals = shap.KernelExplainer(self.predict_func, np.array([outlier_point]))
            local_explanations[o] = det_shap_vals.shap_values(np.array([outlier_point]))
        print(local_explanations)
