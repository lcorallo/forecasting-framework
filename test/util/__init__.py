import matplotlib.pyplot as plt

from .test_suite import TestSuite


def save_performance_graph(
    path,
    id,
    series,
    series_name,
    y_train,
    yhat_train,
    y_test,
    yhat_test,
    rmse_train,
    rmse_test):
    # def showExperiment(ID, mixed, series, series_name, Y_train, Y_test, Yhat_train, Yhat_test, Yhat_series, rmse_train, rmse_test):
    plt.figure(figsize=(30,10))
    plt.subplot(311)
    plt.margins(0.05)
    plt.plot(series)
    # plt.plot(Yhat_series)
    plt.legend(["True", "Predicted"])
    plt.title(series_name)

    plt.subplot(323)
    plt.margins(0.05)
    plt.plot(y_train)
    plt.plot(yhat_train)
    plt.legend(["True", "Predicted"])
    plt.title("RMSE Train: {rmse_train:.5f}")

    plt.subplot(324)
    plt.margins(0.05)
    plt.plot(y_test)
    plt.plot(yhat_test)
    plt.legend(["True", "Predicted"])
    plt.title("RMSE Test: {rmse_test:.5f}")

    plt.subplot(325)
    plt.margins(0.05)
    plt.plot(abs(y_train - yhat_train))
    plt.legend(["Residuals"])
    plt.title("Train")

    plt.subplot(326)
    plt.margins(0.05)
    plt.plot(abs(y_test - yhat_test))
    plt.legend(["Residuals"])
    plt.title("Test")

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    # plt.savefig(path+str(id)+".png", transparent=True)