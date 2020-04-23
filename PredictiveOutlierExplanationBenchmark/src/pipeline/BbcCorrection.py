import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_metric


class BBC:

    __B = 1000

    def __init__(self, y_true, out_of_sample_predictions, metric_id):
        self.out_of_sample_predictions = out_of_sample_predictions
        self.metric_id = metric_id
        self.y_true = y_true

    def correct_bias(self):
        N = self.out_of_sample_predictions.shape[0]
        ids = np.arange(N)
        perf = 0.0
        for i in range(BBC.__B):
            b = np.random.choice(N, N, replace=True)
            b_prime = np.delete(ids, b)
            max_c = self.__configuration_selection_strategy(self.y_true[b], self.out_of_sample_predictions[b, :])
            metric_dict = calculate_metric(self.y_true[b_prime], self.out_of_sample_predictions[b_prime, max_c], self.metric_id)
            key = next(iter(metric_dict))
            val = metric_dict[key]
            perf += val
        return perf / BBC.__B

    def __configuration_selection_strategy(self, y_true, out_of_sample_predictions):
        max_c = None
        max_val = None
        configurations = out_of_sample_predictions.shape[1]
        for c in range(configurations):
            metric_dict = calculate_metric(y_true, out_of_sample_predictions[:, c], self.metric_id)
            key = next(iter(metric_dict))
            val = metric_dict[key]
            if max_val is None or max_val < val:
                max_val = val
                max_c = c
        return max_c
