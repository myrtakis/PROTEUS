import warnings


warnings.filterwarnings("ignore")


class SES:

    def __init__(self, params):
        self.__params = params

    def run(self, X_train, Y_train):
        warnings.warn('SES is removed from this version of PROTEUS to avoid conflicts with Apache 2.0 license. SES returns an empty array of features.')
        return [], None
