from src.models.regressor.interface import IRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

class Model_Linear_SVR(IRegressor):
    model = None;

    def __init__(self, input_size, C=None, epsilon=None, fit_intercept=None):
        self.input_size = input_size
        self.C = C
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.model = LinearSVR(C=C, epsilon=epsilon, fit_intercept=fit_intercept)

    def __train__(self, X_train, Y_train):    
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        return yhat_train

    def __test__(self, X_test):
        yhat_test = self.model.predict(X_test)
        return yhat_test

    def __identify__(self):
        return "AR_SupportVectorLinearRegression("+str(self.input_size)+")("+str(self.C)+","+str(self.epsilon)+","+str(self.fit_intercept)+");"


class Model_RBF_SVR(IRegressor):
    model = None;

    def __init__(self, input_size, C=None, epsilon=None, gamma=None):
        self.input_size = input_size
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)

    def __train__(self, X_train, Y_train):    
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        return yhat_train

    def __test__(self, X_test):
        yhat_test = self.model.predict(X_test)
        return yhat_test

    def __identify__(self):
        return "AR_SupportVectorRBFRegression("+str(self.input_size)+")("+str(self.C)+","+str(self.epsilon)+","+str(self.gamma)+");"


