from sklearn.neighbors import KNeighborsClassifier


class KNN:

    def __init__(self, params):
        self.__params = params
        self.__model = None

    def train(self, X_train, Y_train):
        return KNeighborsClassifier(n_neighbors=self.__params['n_neighbors']).fit(X_train, Y_train)

    def predict(self, X_test):
        return self.__model.predict(X_test)
