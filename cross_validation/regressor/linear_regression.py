import numpy as np

from models.regressor.linear_regression import Model_LinearRegression
from .kfold import KFoldCrossValidation
from performer.supervised_learning_performer import SlidingWindowPerformer
from util.parameters import ModelsIperParameters
from util.evaluation import ForecastErrorEvaluation

class KFoldCrossValidation_LinearRegression(KFoldCrossValidation):

  def __init__(self, series, target_length=1, target_offset=1, kfolds=10):
    self.series = series
    self.kfolds = kfolds

    self.target_length = target_length
    self.target_offset = target_offset

  def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
    features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)

    for ind_feature in features_length:
      SWP = SlidingWindowPerformer(
        feature_length=ind_feature,
        target_length=self.target_length,
        target_offset=self.target_offset
      )

      _, X_train, Y_train = SWP.get(self.series)
      X_trains, X_validations, Y_trains, Y_validations = self._generate_train_validation_sets(X_train, Y_train)

      validation_loss = np.array([])
      for i in range(len(X_trains)):
        y = Y_validations[i]

        model_linear_regression = Model_LinearRegression()
        model_linear_regression.__train__(X_train=X_trains[i], Y_train=Y_trains[i])
        yhat = model_linear_regression.__test__(X_test=X_validations[i])

        loss = error_performer.get(y, yhat)
        validation_loss = np.append(validation_loss, loss)
      
      loss_mean = np.mean(validation_loss, axis=0)
      loss_std = np.std(validation_loss, axis=0)
      self._cross_validation_results.append([
        loss_mean, loss_std,
        {ModelsIperParameters.FEATURE_LENGTH: ind_feature},
        model_linear_regression
      ])