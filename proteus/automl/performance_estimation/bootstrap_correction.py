import numpy as np


class BBC:

    def __init__(self, y_true, out_of_sample_predictions, metric_func, bootstraps=100):
        assert y_true is not None and len(y_true) > 0
        assert out_of_sample_predictions is not None and len(out_of_sample_predictions) > 0
        self.out_of_sample_predictions = out_of_sample_predictions
        self.metric_func = metric_func
        self.y_true = y_true
        self.bootstraps = bootstraps

    def correct_bias(self):
        N = self.out_of_sample_predictions.shape[0]
        ids = np.arange(N)
        out_perf = np.zeros(self.bootstraps)
        for i in range(self.bootstraps):
            print('\r', 'Iteration', '(', i, '/', self.bootstraps, ')',  end='')
            b = np.random.choice(N, N, replace=True)
            b_prime = np.delete(ids, b)
            perfs = []
            for c in self.out_of_sample_predictions.columns:
                if len(np.unique(self.y_true[b])) == 1:
                    perfs.append(-1)
                    continue
                curr_perf = self.metric_func(self.y_true[b], self.out_of_sample_predictions.loc[b, c])
                perfs.append(curr_perf)
            perfs = np.array(perfs)
            max_c = np.argmax(perfs)
            if len(np.unique(self.y_true[b_prime])) == 1:
                out_perf[i] = -1
            else:
                out_perf[i] = self.metric_func(self.y_true[b_prime], self.out_of_sample_predictions.loc[b_prime, max_c])
        invalid_vals = np.where(out_perf == -1)[0]
        if len(invalid_vals) > 0:
            print('\nWarning:', len(invalid_vals), 'iters out of', self.bootstraps, 'contained only one class and omitted')
            out_perf = np.delete(out_perf, invalid_vals)
        conf = 0.95
        a = 0.5 * (1-conf)
        ci = np.quantile(a=out_perf, q=[a, 1-a])
        return np.mean(out_perf), ci
