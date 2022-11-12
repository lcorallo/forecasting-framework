import matplotlib.pyplot as plt

from .test_suite import TestSuite


def save_performance_graph(
        path,
        id,
        series,
        yhat_series,
        series_name,
        y_train,
        yhat_train,
        y_test,
        yhat_test,
        error_train,
        error_test
    ):

    formatted_error_train = f'{error_train:.5f}'
    formatted_error_test = f'{error_test:.5f}'

    plt.figure(figsize=(30,10))
    plt.subplot(311)
    plt.margins(0.02)
    plt.plot(series)
    plt.plot(yhat_series, linestyle='--')
    plt.legend(["True", "Predicted"])
    plt.title(series_name)

    plt.subplot(323)
    plt.margins(0.02)
    plt.plot(y_train)
    plt.plot(yhat_train, linestyle='--', linewidth=1, marker='o', markersize=4)
    plt.legend(["True", "Predicted"])
    plt.title("RMSE Train: "+str(formatted_error_train))

    plt.subplot(324)
    plt.margins(0.02)
    plt.plot(y_test)
    plt.plot(yhat_test, linestyle='--', linewidth=1, marker='o', markersize=4)
    plt.legend(["True", "Predicted"])
    plt.title("RMSE Test: "+str(formatted_error_test))

    plt.subplot(325)
    plt.margins(0.02)
    plt.plot(abs(y_train - yhat_train), linestyle='--', linewidth=1, marker='o', markersize=4)
    plt.legend(["Train Residuals"])

    plt.subplot(326)
    plt.margins(0.02)
    plt.plot(abs(y_test - yhat_test), linestyle='--', linewidth=1, marker='o', markersize=4)
    plt.legend(["Test Residuals"])

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.savefig(path+str(id)+".png", transparent=True)