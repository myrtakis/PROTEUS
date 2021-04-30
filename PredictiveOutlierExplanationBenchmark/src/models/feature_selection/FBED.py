import warnings


class FBED:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        warnings.warn(
            'FBED is removed from this version of PROTEUS to avoid conflicts with Apache 2.0 license. FBED returns an empty array of features.')
        return [], None
