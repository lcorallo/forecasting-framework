import numpy as np
from sklearn.model_selection import KFold

from src.models.regressor.svr import Model_Linear_SVR
from src.models.regressor.svr import Model_RBF_SVR
from src.performer.train_test_performer import TrainTestPerformer

from src.performer.supervised_learning_performer import SlidingWindowPerformer

from src.util.parameters import ModelsIperParameters
from src.util.evaluation import ForecastErrorEvaluation


class GridSearchKFoldCV_Linear_SVR():

  _RANDOM_SEED = 1
  _cross_validation_results = []

  def get_all_params(self):
      cross_validation_results = np.array(self._cross_validation_results)
      return cross_validation_results[:, 2:4]

  def get_best_params(self):
      cross_validation_results = np.array(self._cross_validation_results)
      min = cross_validation_results.T[0].min()
      index_min = np.where(cross_validation_results.T[0] == min)
      return cross_validation_results[index_min][0][0], cross_validation_results[index_min][0][1], cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]

  def _generate_train_validation_sets(self, X_train, Y_train): 
      kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self._RANDOM_SEED)
      new_X_train, new_X_validation, new_Y_train, new_Y_validation = [], [], [], []

      for train_index, test_index in kfold.split(X_train):
          new_X_train.append(X_train[train_index])
          new_Y_train.append(Y_train[train_index])
          new_X_validation.append(X_train[test_index])
          new_Y_validation.append(Y_train[test_index])
      
      return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)
  
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
            TTP = TrainTestPerformer(portion_train = 0.8, random_sampling = True)

            _, X, Y = SWP.get(self.series)
            X_train, _, Y_train, _ = TTP.get(X, Y)
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


class GridSearchKFoldCV_RBF_SVR():

  _RANDOM_SEED = 1
  _cross_validation_results = []

  def get_all_params(self):
      cross_validation_results = np.array(self._cross_validation_results)
      return cross_validation_results[:, 2:4]

  def get_best_params(self):
      cross_validation_results = np.array(self._cross_validation_results)
      min = cross_validation_results.T[0].min()
      index_min = np.where(cross_validation_results.T[0] == min)
      return cross_validation_results[index_min][0][0], cross_validation_results[index_min][0][1], cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]

  def _generate_train_validation_sets(self, X_train, Y_train): 
      kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self._RANDOM_SEED)
      new_X_train, new_X_validation, new_Y_train, new_Y_validation = [], [], [], []

      for train_index, test_index in kfold.split(X_train):
          new_X_train.append(X_train[train_index])
          new_Y_train.append(Y_train[train_index])
          new_X_validation.append(X_train[test_index])
          new_Y_validation.append(Y_train[test_index])
      
      return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)
  
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

            TTP = TrainTestPerformer(portion_train = 0.8, random_sampling = True)

            _, X, Y = SWP.get(self.series)
            X_train, _, Y_train, _ = TTP.get(X, Y)
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