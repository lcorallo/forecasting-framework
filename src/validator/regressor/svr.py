import numpy as np

from src.models.regressor.svr import Model_Linear_SVR
from src.models.regressor.svr import Model_RBF_SVR

from .kfold import KFoldCrossValidation

from src.performer.supervised_learning_performer import SlidingWindowPerformer

from src.util.parameters import ModelsIperParameters
from src.util.evaluation import ForecastErrorEvaluation


class GridSearchKFoldCV_Linear_SVR(KFoldCrossValidation):
  
  def __init__(self, series, target_length=1, target_offset=1, kfolds=10):
    self.series = series
    self.kfolds = kfolds

    self.target_length = target_length
    self.target_offset = target_offset

  def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
    features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)
    c_param = params.get(ModelsIperParameters.C)
    epsilon_param = params.get(ModelsIperParameters.EPSILON)
    fit_intercept_param = params.get(ModelsIperParameters.FIT_INTERCEPT)
    
    for ind_feature in features_length:
      for ind_c in c_param:
        for ind_epsilon in epsilon_param:
          for ind_fit_intercept in fit_intercept_param:
            SWP = SlidingWindowPerformer(
              feature_length=ind_feature,
              target_length=self.target_length,
              target_offset=self.target_offset
            )

            _, X_train, Y_train = SWP.get(self.series)
            Y_train = Y_train.ravel()
            X_trains, X_validations, Y_trains, Y_validations = self._generate_train_validation_sets(X_train, Y_train)

            validation_loss = np.array([])
            for i in range(len(X_trains)):
              y = Y_validations[i]

              model_svr_linear = Model_Linear_SVR(input_size= X_trains[i].shape[1], C = ind_c, epsilon = ind_epsilon, fit_intercept = ind_fit_intercept)
              model_svr_linear.__train__(X_train=X_trains[i], Y_train=Y_trains[i])
              yhat = model_svr_linear.__test__(X_test=X_validations[i])
              
              loss = error_performer.get(y, yhat)
              validation_loss = np.append(validation_loss, loss)
            
            loss_mean = np.mean(validation_loss, axis=0)
            loss_std = np.std(validation_loss, axis=0)
            self._cross_validation_results.append([
              loss_mean,
              loss_std,
              {
                ModelsIperParameters.FEATURE_LENGTH: ind_feature,
                ModelsIperParameters.C: ind_c,
                ModelsIperParameters.EPSILON: ind_epsilon,
                ModelsIperParameters.FIT_INTERCEPT: ind_fit_intercept
              },
              model_svr_linear
            ])


class GridSearchKFoldCV_RBF_SVR(KFoldCrossValidation):
  
  def __init__(self, series, target_length=1, target_offset=1, kfolds=10):
    self.series = series
    self.kfolds = kfolds

    self.target_length = target_length
    self.target_offset = target_offset

  def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
    features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)
    c_param = params.get(ModelsIperParameters.C)
    epsilon_param = params.get(ModelsIperParameters.EPSILON)
    gamma_param = params.get(ModelsIperParameters.GAMMA)
    
    for ind_feature in features_length:
      for ind_c in c_param:
        for ind_epsilon in epsilon_param:
          for ind_gamma in gamma_param:
            SWP = SlidingWindowPerformer(
              feature_length=ind_feature,
              target_length=self.target_length,
              target_offset=self.target_offset
            )

            _, X_train, Y_train = SWP.get(self.series)
            Y_train = Y_train.ravel()
            X_trains, X_validations, Y_trains, Y_validations = self._generate_train_validation_sets(X_train, Y_train)

            validation_loss = np.array([])
            for i in range(len(X_trains)):
              y = Y_validations[i]

              model_svr_rbf = Model_RBF_SVR(input_size= X_trains[i].shape[1], C = ind_c, epsilon = ind_epsilon, gamma = ind_gamma)
              model_svr_rbf.__train__(X_train=X_trains[i], Y_train=Y_trains[i])
              yhat = model_svr_rbf.__test__(X_test=X_validations[i])
              
              loss = error_performer.get(y, yhat)
              validation_loss = np.append(validation_loss, loss)
            
            loss_mean = np.mean(validation_loss, axis=0)
            loss_std = np.std(validation_loss, axis=0)
            self._cross_validation_results.append([
              loss_mean,
              loss_std,
              {
                ModelsIperParameters.FEATURE_LENGTH: ind_feature,
                ModelsIperParameters.C: ind_c,
                ModelsIperParameters.EPSILON: ind_epsilon,
                ModelsIperParameters.GAMMA: ind_gamma
              },
              model_svr_rbf
            ])