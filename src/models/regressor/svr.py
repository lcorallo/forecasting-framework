from models.regressor.IRegressor import IRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

class Model_Linear_SVR(IRegressor):
    model = None;

    def __init__(self, C=None, epsilon=None, fit_intercept=None):
        self.model = LinearSVR(C=C, epsilon=epsilon, fit_intercept=fit_intercept)

    def __train__(self, X_train, Y_train):    
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        return yhat_train

    def __test__(self, X_test):
        yhat_test = self.model.predict(X_test)
        return yhat_test


class Model_RBF_SVR(IRegressor):
    model = None;

    def __init__(self, C=None, epsilon=None, gamma=None):
        self.model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)

    def __train__(self, X_train, Y_train):    
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        return yhat_train

    def __test__(self, X_test):
        yhat_test = self.model.predict(X_test)
        return yhat_test


