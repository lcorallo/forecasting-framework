from src.performer.transformer import MinMaxTransformer
from src.pipeline import IPipeline
from src.util.evaluation import ForecastErrorEvaluation
from src.util.goal import ForecastingGoal
from src.util.parameters import ModelsIperParameters, NeuralNetworkTrainingParameters
from src.validator.nn.trainer import GridSearch_ConvolutionalNeuralNetwork


class Use_ARConvolutionalNeuralNetwork(IPipeline):
    
    def __execute__(self, series, goal: ForecastingGoal):
        CNN_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[3,4,5,6,7,8,9,10]
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