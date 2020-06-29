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

    @abc.abstractmethod
    def calculate_explanation(self, outlier_ids):
        """If the detector is explainable return the explanation"""
        return None

    @abc.abstractmethod
    def get_explanation(self):
        return None

    @abc.abstractmethod
    def is_explainable(self):
        return False
