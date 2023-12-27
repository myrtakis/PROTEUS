import abc


class BaseFeatureSelector:

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("fit method should be implemented")

    @abc.abstractmethod
    def feature_coefs(self, X, dim):
        raise NotImplementedError("feature_coefs method should be implemented")

