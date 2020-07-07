import numpy as np


class RandomSelector:

    def __init__(self, dataset_detected_outliers):
        self.dataset = dataset_detected_outliers

    def run(self):
        return np.random.normal(0, 1, self.dataset.get_X().shape[1])
