from src.models.regressor.interface import IRegressor
from sklearn.linear_model import LinearRegression

#TODO: One Step Forecasting               V
#TODO: One Step Forecasting With Offset   V
#TODO: One Shot Forecasting               V
#TODO: One Shot Forecasting With Offset   V
#TODO: Multi Step Forecasting             X
#TODO: Multi Step Forecasting With Offset X
class Model_LinearRegression(IRegressor):
    model = None;

    def __init__(self):
        self.model = LinearRegression()

    def __get_model__(self):
        return self.model

    def __train__(self, X_train, Y_train):
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        return yhat_train

    def __test__(self, X_test):
        yhat_test = self.model.predict(X_test)
        return yhat_test

