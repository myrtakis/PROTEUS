import abc


class BaseClassifier:

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("fit method should be implemented")

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError("predict method should be implemented")
