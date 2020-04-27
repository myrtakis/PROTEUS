import numpy as np
from PredictiveOutlierExplanationBenchmark.src.utils.metrics import calculate_metric


class BBC:

    __B = 1000
    __cores = 4

    def __init__(self, y_true, out_of_sample_predictions, metric_id):
        self.out_of_sample_predictions = out_of_sample_predictions
        self.metric_id = metric_id
        self.y_true = y_true

    def correct_bias(self):
        import multiprocessing as mp
        pool = mp.Pool(BBC.__cores)
        result = pool.map(self.correct_parallel, ((self.y_true, self.out_of_sample_predictions, self.metric_id, i) for i in range(BBC.__B)))
        pool.close()
        pool.join()
        assert len(result) == BBC.__B, str(len(result)) + ' != ' + str(BBC.__B)
        result = np.array(result)
        invalid_vals = np.where(result == -1)[0]
        if len(invalid_vals) > 0:
            print(' Warning:', len(invalid_vals), 'iters out of', BBC.__B, 'contained only one class and omitted', end='')
            result = np.delete(result, invalid_vals)
        return np.mean(result)

    def correct_parallel(self, v):
        y_true, out_of_sample_predictions, metric_id, i = v
        N = out_of_sample_predictions.shape[0]
        ids = np.arange(N)
        perf = 0.0
        print('\r', 'Removing bias for metric', self.metric_id, '(', i, '/', BBC.__B, ')',  end='')
        b = np.random.choice(N, N, replace=True)
        b_prime = np.delete(ids, b)
        max_c = None
        max_val = None
        configurations = out_of_sample_predictions.shape[1]
        for c in range(configurations):
            metric_dict = calculate_metric(y_true, out_of_sample_predictions[:, c], metric_id)
            if max_val is None or max_val < metric_dict[metric_id]:
                max_val = metric_dict[metric_id]
                max_c = c
        metric_dict = calculate_metric(y_true[b_prime], out_of_sample_predictions[b_prime, max_c],
                                       metric_id)
        key = next(iter(metric_dict))
        val = metric_dict[key]
        perf += val
        return perf

    def __configuration_selection_strategy(self, y_true, out_of_sample_predictions):
        max_c = None
        max_val = None
        configurations = out_of_sample_predictions.shape[1]
        for c in range(configurations):
            metric_dict = calculate_metric(y_true, out_of_sample_predictions[:, c], self.metric_id)
            if max_val is None or max_val < metric_dict[self.metric_id]:
                max_val = metric_dict[self.metric_id]
                max_c = c
        return max_c
