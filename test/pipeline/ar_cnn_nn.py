import pandas as pd
import numpy as np
import torch
from src.models.nn.cnn import CNNSemilinearPredictor

from src.performer.supervised_learning_performer import SlidingWindowPerformer
from src.performer.train_test_performer import TrainTestPerformer
from src.util.goal import OneStepForecastingGoal
from src.util.evaluation import ForecastErrorEvaluation
from src.util.parameters import ModelsIperParameters
from src.performer.transformer import MinMaxTransformer

from src.validator.nn.trainer import NeuralNetworkTrainingParameters, Trainer_NeuralNetwork
from src.validator.nn.trainer import GridSearch_ConvolutionalNeuralNetwork

from src.pipeline import IPipeline
from test.util import save_performance_graph
from test.util.test_suite import TestSuite
       
class Test_ARConvolutionalNeuralNetwork(IPipeline):
    def __test_execute__(self):
        test_suite = TestSuite()
        DIR = "test/output/ar_cnn_nn2/"
        FILE = "data.csv"
        OUTPUT = DIR + FILE

        csv_output = pd.DataFrame(columns=[
            "ID", "Series", "Target Length", "Target Offset", "AR Features", "Residuals Train", "Error Train", "Residuals Test", "Error Test"
        ])
        CNN_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4,5,6,7,8,9,10,15,20]
        )
        training_parameters = NeuralNetworkTrainingParameters(EPOCHS = 900, EARLY_STOP = True, LEARNING_RATE = .001)

        #Goal Offset
        # TARGET_OFFSET = [1,2,3,4,5,11,12,13,14,15,21,22,23,24,25]
        TARGET_OFFSET = [1,2,3,4,5]
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

                #Search Best AR Parameters
                grid_searcher = GridSearch_ConvolutionalNeuralNetwork(
                    series = series,
                    target_length = forecast_view,
                    target_offset = forecast_offset,
                    training_parameters = training_parameters
                )
                grid_searcher.search(CNN_IPER_PARAMETERS, ERROR_PERFORMER)
                _, parameters, _ = grid_searcher.get_best_params()

                #Creating Model From Best parameters
                SWP = SlidingWindowPerformer(
                    feature_length=parameters[CNN_IPER_PARAMETERS.FEATURE_LENGTH],
                    target_length=1,
                    target_offset=forecast_offset
                )

                TTP = TrainTestPerformer(portion_train = 0.8, random_sampling = True)
                _, X, Y = SWP.get(series)

                X_train, X_test, Y_train, Y_test = TTP.get(X, Y)

                X_train = torch.tensor(data = X_train, dtype = torch.float32)
                Y_train = torch.tensor(data = Y_train, dtype = torch.float32)
                X_test = torch.tensor(data = X_test, dtype = torch.float32)
                Y_test = torch.tensor(data = Y_test, dtype = torch.float32)

                model = CNNSemilinearPredictor(
                    n_inp = parameters[CNN_IPER_PARAMETERS.FEATURE_LENGTH],
                    n_out = 1,
                    conv1_out = 6,
                    conv1_kernel = 36,
                    conv2_kernel = 12,
                    layers_fc = 2,
                    linear = 180
                )
                        
                trainer = Trainer_NeuralNetwork(training_parameters = training_parameters)
                model, _ = trainer.__train__(model = model, x = X_train, y = Y_train, error_performer = ERROR_PERFORMER)

                model.eval()
                with torch.no_grad():
                    yhat_train = model(X_train)
                    error_train = ERROR_PERFORMER.get(Y_train, yhat_train)

                model.eval()
                with torch.no_grad():
                    yhat_test = model(X_test)
                    error_test = ERROR_PERFORMER.get(Y_test, yhat_test)

                #Only for plotting:
                model.eval()
                with torch.no_grad():
                    X = torch.tensor(data = X, dtype = torch.float32)
                    yhat_series = model(X)

                Y_train = Y_train.detach().numpy()
                yhat_train = yhat_train.detach().numpy()
                Y_test = Y_test.detach().numpy()
                yhat_test = yhat_test.detach().numpy()
                yhat_series = yhat_series.detach().numpy()
                new_row = {
                    'ID': testIdAutoincrement,
                    'Series': series_name,
                    "Target Length": 1,
                    "Target Offset": ind_target_of,
                    'AR Features': parameters[CNN_IPER_PARAMETERS.FEATURE_LENGTH],
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

