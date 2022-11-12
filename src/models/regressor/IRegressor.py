class IRegressor():

    def __train__(self, X_train, Y_train):
        raise ValueError("Method not implemented")

    def __test__(self, X_test, Y_test):
        raise ValueError("Method not implemented")