from models.IModel import IModel
from statsmodels.tsa.arima.model import ARIMA


class Model_ARIMA(IModel):
    error_train = None;
    error_test = None;
    model = None;
    cross_validator = None;
    error_fun = None;

    def __init__(self, error_fun):
        self.error_fun = error_fun
        self.model = SVR(kernel="rbf")
        self.cross_validator = GridSearchCV(
            estimator=self.model,
            param_grid={
                'C': [0.05, 0.1, 1, 2, 4, 8, 10],
                'epsilon': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5],
                'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5]
            },
            cv=10,
            n_jobs=-1,
            scoring=make_scorer(error_fun, greater_is_better=False)
        )

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
        grid_result = self.cross_validator.fit(X_train, Y_train)
        best_params = grid_result.best_params_

        self.model = SVR(kernel="rbf", C=best_params['C'], epsilon=best_params['epsilon'], gamma=best_params['gamma'])
        self.model = self.model.fit(X_train, Y_train)
        yhat_train = self.model.predict(X_train)
        self.error_train = self.error_fun(Y_train, yhat_train)
        return yhat_train

    def __test__(self, X_test, Y_test):
        yhat_test = self.model.predict(X_test)
        self.error_test = self.error_fun(Y_test, yhat_test)
        return yhat_test