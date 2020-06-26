import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_metric


class BBC:

    __B = 1000

    def __init__(self, y_true, out_of_sample_predictions, metric_id):
        assert y_true is not None and len(y_true) > 0
        assert out_of_sample_predictions is not None and len(out_of_sample_predictions) > 0
        self.out_of_sample_predictions = out_of_sample_predictions
        self.metric_id = metric_id
        self.y_true = y_true
        self.repeated_predictions = len(out_of_sample_predictions) > 1

    def correct_bias(self):
        N = self.out_of_sample_predictions[0].shape[0]
        ids = np.arange(N)
        out_perf = np.zeros(BBC.__B)
        for i in range(BBC.__B):
            print('\r', 'Removing bias for metric', self.metric_id, '(', i, '/', BBC.__B, ')',  end='')
            b = np.random.choice(N, N, replace=True)
            b_prime = np.delete(ids, b)
            # the run_R parameter has effect when true only for roc auc metric as it will be calculated from Rfast package in R
            perfs = {}
            bootstrap_is_valid = True
            for preds in self.out_of_sample_predictions:
                curr_perf = calculate_metric(self.y_true[b], preds[b, :], self.metric_id, run_R=True)[self.metric_id]
                if isinstance(curr_perf, int):
                    assert curr_perf == -1
                    bootstrap_is_valid = False
                    break
                curr_perf = np.array(curr_perf)
                if len(perfs) == 0:
                    perfs[self.metric_id] = curr_perf
                else:
                    perfs[self.metric_id] += curr_perf
            if not bootstrap_is_valid:
                out_perf[i] = -1
            else:
                print('\n****PRINTING***\n', perfs)
                perfs = list(perfs[self.metric_id])
                max_c = np.argmax(perfs)
                best_test_perf = {}
                for preds in self.out_of_sample_predictions:
                    curr_perf = np.array(calculate_metric(self.y_true[b_prime], preds[b_prime, max_c], self.metric_id, run_R=True)[self.metric_id])
                    if len(best_test_perf) == 0:
                        best_test_perf[self.metric_id] = curr_perf
                    else:
                        best_test_perf[self.metric_id] += curr_perf
                out_perf[i] = best_test_perf[self.metric_id] / float(len(self.out_of_sample_predictions))
        invalid_vals = np.where(out_perf == -1)[0]
        if len(invalid_vals) > 0:
            print('\nWarning:', len(invalid_vals), 'iters out of', BBC.__B, 'contained only one class and omitted')
            out_perf = np.delete(out_perf, invalid_vals)
        conf = 0.95
        a = 0.5 * (1-conf)
        ci = np.quantile(a=out_perf, q=[a, 1-a])
        return np.mean(out_perf), ci
