from models.IModel import IModel
from sklearn.linear_model import LinearRegression
import numpy as np

#TODO: One Step Forecasting               V
#TODO: One Step Forecasting With Offset   V
#TODO: One Shot Forecasting               V
#TODO: One Shot Forecasting With Offset   V
#TODO: Multi Step Forecasting             X
#TODO: Multi Step Forecasting With Offset X
class Model_LinearRegression(IModel):
    error_train = None;
    error_test = None;
    model = None;
    error_fun = None;

    def __init__(self, error_fun):
        self.error_fun = error_fun
        self.model = LinearRegression()

    def __get_error_train__(self):
        if(self.error_train is None):
            raise Exception('Sorry, train error train isn\'t set yet')
        return self.error_train.astype(np.float32)

    def __get_error_test__(self):
        if(self.error_test is None):
            raise Exception('Sorry, test error train isn\'t set yet')
        return self.error_test.astype(np.float32)

    def __get_model__(self):
        return self.model

    def __get_yhat_train__(self):
        return self.yhat_train

    def __get_yhat_test__(self):
        return self.yhat_test

    def __train__(self, X_train, Y_train):
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        self.error_train = self.error_fun(Y_train, yhat_train)
        return yhat_train

    def __test__(self, X_test, Y_test):
        yhat_test = self.model.predict(X_test)
        self.error_test = self.error_fun(Y_test, yhat_test)
        return yhat_test

