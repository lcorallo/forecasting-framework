from src.performer.transformer import MinMaxTransformer
from src.pipeline import IPipeline
from src.util.evaluation import ForecastErrorEvaluation
from src.util.goal import ForecastingGoal
from src.util.parameters import ModelsIperParameters
from src.validator.regressor.linear_regression import GridSearchKFoldCV_LinearRegression


class Use_ARLinearRegression(IPipeline):
    def __execute__(self, series, goal: ForecastingGoal):
        LINEAR_REGRESSION_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[5]
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