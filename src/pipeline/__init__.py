from src.util.goal import ForecastingGoal
from src.util.evaluation import ForecastErrorEvaluation
from src.util.parameters import ModelsIperParameters
from src.performer.transformer import MinMaxTransformer

from src.validator.regressor.linear_regression import GridSearchKFoldCV_LinearRegression
from src.validator.regressor.svr import GridSearchKFoldCV_Linear_SVR
from src.validator.regressor.svr import GridSearchKFoldCV_RBF_SVR
from src.validator.nn.trainer import NeuralNetworkTrainingParameters
from src.validator.nn.trainer import GridSearch_ConvolutionalNeuralNetwork
from src.validator.nn.trainer import GridSearch_LongShortTermNeuralNetwork

class IPipeline():
    def __execute__(self, series, goal: ForecastingGoal):
        raise ValueError("Method not implemented")
    
    def __test_execute__(self):
        raise ValueError("Method not implemented")


class Use_ARLinearRegression(IPipeline):
    def __execute__(self, series, goal: ForecastingGoal):
        LINEAR_REGRESSION_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4,5,6,7,8,9,10,15,20]
        )
        series = MinMaxTransformer.transform(series)

        ERROR_PERFORMER = ForecastErrorEvaluation(goal = goal)
        forecast_view, forecast_offset = goal.options()

        grid_searcher = GridSearchKFoldCV_LinearRegression(
            series = series,
            target_length = forecast_view,
            target_offset = forecast_offset,
            kfolds = 10
        )
        grid_searcher.search(LINEAR_REGRESSION_IPER_PARAMETERS, ERROR_PERFORMER)
        loss, _, parameters, model = grid_searcher.get_best_params()
        return model, parameters, loss

class Use_ARSupportVectorRegressionRBF(IPipeline):
    def __execute__(self, series, goal:ForecastingGoal):
        # SVR_RBF_IPER_PARAMETERS = ModelsIperParameters(
        #     FEATURE_LENGTH=[3,4,5,6,7,8,9,10,15,20],
        #     C = [0.05, 0.1, 1, 2, 4, 8, 10],
        #     EPSILON = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        #     GAMMA =  [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]
        # )
        SVR_RBF_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4],
            C = [0.05],
            EPSILON = [1e-6],
            GAMMA =  [1e-4]
        )

        series = MinMaxTransformer.transform(series)

        ERROR_PERFORMER = ForecastErrorEvaluation(goal = goal)
        forecast_view, forecast_offset = goal.options()

        grid_searcher = GridSearchKFoldCV_RBF_SVR(
            series = series,
            target_length = forecast_view,
            target_offset = forecast_offset,
            kfolds = 10
        )
        grid_searcher.search(SVR_RBF_IPER_PARAMETERS, ERROR_PERFORMER)
        loss, _, parameters, model =  grid_searcher.get_best_params()
        return model, parameters, loss
        
class Use_ARSupportVectorRegressionLinear(IPipeline):
    def __execute__(self, series, goal:ForecastingGoal):
        # SVR_RBF_IPER_PARAMETERS = ModelsIperParameters(
        #     FEATURE_LENGTH=[3,4,5,6,7,8,9,10,15,20],
        #     C = [0.05, 0.1, 1, 2, 4, 8, 10],
        #     EPSILON = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        #     FIT_INTERCEPT =  [True, False]
        # )
        SVR_LINEAR_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4],
            C = [0.05],
            EPSILON = [1e-6],
            FIT_INTERCEPT =  [True]
        )

        series = MinMaxTransformer.transform(series)

        ERROR_PERFORMER = ForecastErrorEvaluation(goal = goal)
        forecast_view, forecast_offset = goal.options()

        grid_searcher = GridSearchKFoldCV_Linear_SVR(
            series = series,
            target_length = forecast_view,
            target_offset = forecast_offset,
            kfolds = 10
        )
        grid_searcher.search(SVR_LINEAR_IPER_PARAMETERS, ERROR_PERFORMER)
        loss, _, parameters, model =  grid_searcher.get_best_params()
        return model, parameters, loss
       
class Use_ARConvolutionalNeuralNetwork(IPipeline):
    
    def __execute__(self, series, goal: ForecastingGoal):
        CNN_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH = [5],
            LINEAR_LAYERS = [1],
            NEURONS = [128],
            DROPOUT = [0.5]
        )
        series = MinMaxTransformer.transform(series)
        
        ERROR_PERFORMER = ForecastErrorEvaluation(goal = goal)
        forecast_view, forecast_offset = goal.options()
        
        grid_searcher = GridSearch_ConvolutionalNeuralNetwork(
            series = series,
            target_length = forecast_view,
            target_offset = forecast_offset,
            training_parameters = NeuralNetworkTrainingParameters(EPOCHS = 900, EARLY_STOP = True, LEARNING_RATE = .001)
        )

        grid_searcher.search(CNN_IPER_PARAMETERS, ERROR_PERFORMER)
        loss, parameters, model = grid_searcher.get_best_params()
        return model, parameters, loss

class Use_ARLongShortTermMemoryNeuralNetwork(IPipeline):

    def __execute__(self, series, goal: ForecastingGoal):
        LSTM_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH = [5],
            HIDDEN_STATE=[128],
            RECURRENT_LAYERS=[2],
            DROPOUT=[0.2]
        )

        series = MinMaxTransformer.transform(series)

        ERROR_PERFORMER = ForecastErrorEvaluation(goal = goal)
        forecast_view, forecast_offset = goal.options()
        
        grid_searcher = GridSearch_LongShortTermNeuralNetwork(
            series = series,
            target_length = forecast_view,
            target_offset = forecast_offset,
            training_parameters = NeuralNetworkTrainingParameters(EPOCHS = 900, EARLY_STOP = True, LEARNING_RATE = .001)
        )

        grid_searcher.search(LSTM_IPER_PARAMETERS, ERROR_PERFORMER)
        loss, parameters, model = grid_searcher.get_best_params()
        return model, parameters, loss
