import abc


class BaseDetector(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, X_train, params):
        pass

    @abc.abstractmethod
    def score_samples(self):
        pass

    @abc.abstractmethod
    def predict_scores(self, new_samples):
        pass

    def get_explanation(self):
        """If the detector is explainable return the explanation"""
        return None