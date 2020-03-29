from sklearn.svm import SVC


class SVM:

    def __init__(self, params):
        self.__params = params
        self.__model = None

    def train(self, X_train, Y_train):
        if str(self.__params['gamma']).lower() == 'na':
            self.__params['gamma'] = 'auto'
        if str(self.__params['degree']).lower() == 'na':
            self.__params['degree'] = 0
        return SVC(gamma=self.__params['gamma'], kernel=self.__params['kernel'], C=self.__params['C'],
                   degree=self.__params['degree']).fit(X_train, Y_train)

    def predict(self, X_test):
        return self.__model.predict(X_test)
