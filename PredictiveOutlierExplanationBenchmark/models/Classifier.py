from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Classifier:

    def __init__(self, classifier_obj):
        self.__classifiers = {
            'rf': {'train': self.__train_random_forest, 'predict': self.__predict_random_forest},
            'svm': {'train': self.__train_svm, 'predict': self.__predict_svm},
            'knn': {'train': self.__train_knn, 'predict': self.__predict_knn}
        }
        assert classifier_obj['id'] in self.__classifiers
        self.__classifier_obj = classifier_obj
        self.__model = None
        self.__predictions = None

    # Base Functions

    def train(self, X_train, Y_train):
        train_func = self.__classifiers[self.__classifier_obj['id']]['train']
        self.__model = train_func(self.__classifier_obj['params'], X_train, Y_train)

    def predict(self, X_test):
        assert self.__model is not None
        predict_func = self.__classifiers[self.__classifier_obj['id']]['predict']
        self.__predictions = predict_func(X_test)

    # Model Train

    def __train_random_forest(self, params, X_train, Y_train):
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      min_samples_leaf=params['min_samples_leaf'],
                                      criterion=params['criterion']).fit(X_train, Y_train)

    def __train_knn(self, params, X_train, Y_train):
        return KNeighborsClassifier(n_neighbors=params['n_neighbors']).fit(X_train, Y_train)

    def __train_svm(self, params, X_train, Y_train):
        if str(params['gamma']).lower() == 'na':
            params['gamma'] = 'auto'
        if str(params['degree']).lower() == 'na':
            params['degree'] = 0
        return SVC(gamma=params['gamma'], kernel=params['kernel'], C=params['C'],
                   degree=params['degree']).fit(X_train, Y_train)

    # Models Predict

    def __predict_random_forest(self, X_test):
        return self.__model.predict(X_test)

    def __predict_svm(self, X_test):
        return self.__model.predict(X_test)

    def __predict_knn(self, X_test):
        return self.__model.predict(X_test)

    # Getter Functions

    def get_predictions(self):
        return self.__predictions
