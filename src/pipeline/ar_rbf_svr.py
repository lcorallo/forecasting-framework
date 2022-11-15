from src.performer.transformer import MinMaxTransformer
from src.pipeline import IPipeline
from src.util.evaluation import ForecastErrorEvaluation
from src.util.goal import ForecastingGoal
from src.util.parameters import ModelsIperParameters
from src.validator.regressor.svr import GridSearchKFoldCV_RBF_SVR


class Use_ARSupportVectorRegressionRBF():
    def __execute__(self, series, goal:ForecastingGoal):
        # SVR_RBF_IPER_PARAMETERS = ModelsIperParameters(
        #     FEATURE_LENGTH=[3,4,5,6,7,8,9,10,15,20],
        #     C = [0.05, 0.1, 1, 2, 4, 8, 10],
        #     EPSILON = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        #     GAMMA =  [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]
        # )
        SVR_RBF_IPER_PARAMETERS = ModelsIperParameters(
            FEATURE_LENGTH=[5],
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