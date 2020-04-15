from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, params):
        self.__params = params
        self.__model = None

    def train(self, X_train, Y_train):
        return RandomForestClassifier(n_estimators=self.__params['n_estimators'],
                                      min_samples_leaf=self.__params['min_samples_leaf'],
                                      criterion=self.__params['criterion']).fit(X_train, Y_train)

    def predict_proba(self, X_test):
        return self.__model.predict_proba(X_test)
