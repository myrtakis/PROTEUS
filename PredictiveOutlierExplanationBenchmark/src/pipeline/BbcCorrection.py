import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_metric


class BBC:

    __B = 1000
    __cores = 2

    def __init__(self, y_true, out_of_sample_predictions, metric_id):
        self.out_of_sample_predictions = out_of_sample_predictions
        self.metric_id = metric_id
        self.y_true = y_true

    def correct_bias(self):
        N = self.out_of_sample_predictions.shape[0]
        ids = np.arange(N)
        out_perf = np.zeros(BBC.__B)
        for i in range(BBC.__B):
            print('\r', 'Removing bias for metric', self.metric_id, '(', i, '/', BBC.__B, ')',  end='')
            b = np.random.choice(N, N, replace=True)
            b_prime = np.delete(ids, b)
            # the run_R parameter has effect when true only for roc auc metric as it will be calculated from Rfast package in R
            perfs = calculate_metric(self.y_true[b], self.out_of_sample_predictions[b, :], self.metric_id, run_R=True)
            perfs = list(perfs[self.metric_id])
            max_c = np.argmax(perfs)
            best_test_perf = calculate_metric(self.y_true[b_prime], self.out_of_sample_predictions[b_prime, max_c],
                                         self.metric_id, run_R=True)
            out_perf[i] = best_test_perf[self.metric_id]
        invalid_vals = np.where(out_perf == -1)[0]
        if len(invalid_vals) > 0:
            print(' Warning:', len(invalid_vals), 'iters out of', BBC.__B, 'contained only one class and omitted',
                  end='')
            out_perf = np.delete(out_perf, invalid_vals)
        return np.mean(out_perf)
