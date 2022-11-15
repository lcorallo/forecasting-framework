import pandas as pd
import numpy as np

from src.models.regressor.linear_regression import Model_LinearRegression
from src.performer.supervised_learning_performer import SlidingWindowPerformer
from src.performer.train_test_performer import TrainTestPerformer
from src.util.goal import OneStepForecastingGoal
from src.util.evaluation import ForecastErrorEvaluation
from src.util.parameters import ModelsIperParameters
from src.performer.transformer import MinMaxTransformer

from src.validator.regressor.linear_regression import GridSearchKFoldCV_LinearRegression

from src.pipeline import IPipeline
from test.util import save_performance_graph
from test.util.test_suite import TestSuite

class Test_ARLinearRegression(IPipeline):
    def __test_execute__(self):
        test_suite = TestSuite()
        DIR = "test/output/ar_linear_regression/"
        FILE = "data.csv"
        OUTPUT = DIR + FILE

        csv_output = pd.DataFrame(columns=[
            "ID", "Series", "Target Length", "Target Offset", "AR Features", "Residuals Train", "Error Train", "Residuals Test", "Error Test"
        ])

        #Model Iper-parameters
        LINEAR_REGRESSION_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4,5,6,7,8,9,10]
        )

        #Goal Offset
        TARGET_OFFSET = [1,2,3,4,5,11,12,13,14,15,21,22,23,24,25]
        number_of_experiment = test_suite.__get_test_suite_size__()

        testIdAutoincrement = 0
        for ind_experiment in range(number_of_experiment):
            series = test_suite.__get_numpy_test_series_from_index__(ind_experiment)
            series = MinMaxTransformer.transform(series)
            series_name = test_suite.__get_name_test_series_from_index__(ind_experiment)

            for ind_target_of in TARGET_OFFSET:
                GOAL = OneStepForecastingGoal(ind_target_of)
                forecast_view, forecast_offset = GOAL.options()
                ERROR_PERFORMER = ForecastErrorEvaluation(goal = GOAL)

                #GridSearchCV Best Parameters
                grid_searcher = GridSearchKFoldCV_LinearRegression(
                    series = series,
                    target_length = forecast_view,
                    target_offset = forecast_offset,
                    kfolds = 10
                )
                grid_searcher.search(LINEAR_REGRESSION_IPER_PARAMETERS, ERROR_PERFORMER)
                _, _, parameters, _ = grid_searcher.get_best_params()

                #Creating Model from best parameters
                SWP = SlidingWindowPerformer(
                    feature_length = parameters[LINEAR_REGRESSION_IPER_PARAMETERS.FEATURE_LENGTH],
                    target_length = 1,
                    target_offset = ind_target_of
                )
                _, X, Y = SWP.get(series)
                TTP = TrainTestPerformer(portion_train = .8, random_sampling=True)
                X_train, X_test, Y_train, Y_test = TTP.get(X, Y)

                model_linear_regression = Model_LinearRegression(input_size = X_train.shape[1])
                yhat_train = model_linear_regression.__train__(X_train, Y_train)
                yhat_test = model_linear_regression.__test__(X_test)
                
                error_train = ERROR_PERFORMER.get(Y_train, yhat_train)
                error_test = ERROR_PERFORMER.get(Y_test, yhat_test)

                #Only for plotting:
                yhat_series = model_linear_regression.__test__(X)

                new_row = {
                    'ID': testIdAutoincrement,
                    'Series': series_name,
                    "Target Length": 1,
                    "Target Offset": ind_target_of,
                    'AR Features': parameters[LINEAR_REGRESSION_IPER_PARAMETERS.FEATURE_LENGTH],
                    'Residuals Train': np.mean(np.abs(Y_train - yhat_train)),
                    'Error Train': error_train,
                    "Residuals Test": np.mean(np.abs(Y_test - yhat_test)),
                    'Error Test': error_test
                }
                csv_output = csv_output.append(new_row, ignore_index=True)
                        
                save_performance_graph(
                    path = DIR,
                    id = testIdAutoincrement,
                    series = Y,
                    yhat_series = yhat_series,
                    y_train = Y_train,
                    y_test = Y_test,
                    series_name = series_name,
                    error_train = error_train,
                    error_test = error_test,
                    yhat_train = yhat_train,
                    yhat_test = yhat_test
                )
                testIdAutoincrement += 1
        csv_output.to_csv(OUTPUT)     