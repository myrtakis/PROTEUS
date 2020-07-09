from collections import OrderedDict
from baselines.micencova import CA_Lasso
import pandas as pd
from models.detectors import *
from holders.Dataset import Dataset
from baselines.shapley import SHAP
from baselines.random import RandomSelector
import numpy as np
import time
import copy


class ExplanationMethods:

    def __init__(self, dataset, detector=None):
        self.dataset = dataset
        self.detector = copy.deepcopy(detector)

    def run_all_post_hoc_explanation_methods(self):
        return {
            'micencova': self.micencova_explanation(),
            'shap': self.shap_explanation(),
            'random': self.random_explanation()
        }

    def micencova_explanation(self):
        print('> Running Micencova explanation baseline')
        output = {}
        avg_feature_scores_reps = np.zeros(self.dataset.get_X().shape[1])
        local_explanations_as_df = None
        repetitions = 10
        start = time.time()
        for i in range(repetitions):
            cal = CA_Lasso(self.dataset)
            local_explanations = cal.run()
            local_explanations_to_df = pd.DataFrame(local_explanations).transpose()
            if local_explanations_as_df is None:
                local_explanations_as_df = local_explanations_to_df
            else:
                local_explanations_as_df = local_explanations_as_df.add(local_explanations_to_df)
            avg_feature_scores_reps += local_explanations_to_df.mean(axis=0).values
        output['time'] = (time.time() - start) / repetitions
        output['global_explanation'] = avg_feature_scores_reps.tolist()
        output['local_explanation'] = OrderedDict(zip(local_explanations_as_df.index, local_explanations_as_df.values.tolist()))
        return output

    def shap_explanation(self):
        print('Running SHAP values explanation baseline')
        assert self.detector is not None
        output = {}
        start = time.time()
        self.detector.train(self.dataset.get_X())
        shap = SHAP(self.dataset, self.detector.predict)
        local_explanations = shap.run()
        output['time'] = time.time() - start
        global_explanation = local_explanations.mean(axis=0)
        output['global_explanation'] = global_explanation.tolist()
        output['local_explanation'] = OrderedDict(zip(local_explanations.index, local_explanations.values.tolist()))
        return output

    def random_explanation(self):
        print('Running Random explanation')
        output = {}
        rs = RandomSelector(self.dataset)
        global_expl = rs.run()
        output['time'] = 0.
        output['global_explanation'] = global_expl.tolist()
        output['local_explanation'] = None
        return output


if __name__ == '__main__':
    df = pd.read_csv('../results_normal/random_oversampling/lof/classification/datasets/synthetic/hics/group_g1/hics_100_g1_001/pseudo_samples_0/pseudo_samples_0_data.csv')
    # df = pd.read_csv('../datasets/synthetic/hics/group_g1/hics_20_g1.csv')
    dataset = Dataset(df, 'is_anomaly', 'subspaces')

    ExplanationMethods(dataset, None).micencova_explanation()

    a = Lof()
    a.train(dataset.get_X(), {'n_neighbors':15})
    # ExplanationMethods(dataset, a).shap_explanation()
