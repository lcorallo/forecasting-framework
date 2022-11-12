from sklearn.model_selection import KFold
import numpy as np

from synthetic.test_suite import TestSuite
from synthetic.util import min_max_transform
from models.regressor.svr import Model_Linear_SVR
from models.regressor.svr import Model_RBF_SVR

class KFoldCrossValidation_Linear_SVR():
  __RANDOM_SEED = 1

  def __init__(self, series, loss, one_step_offset_target, kfolds=10):
    self.series = series
    self.loss = loss
    self.target = one_step_offset_target
    self.kfolds = kfolds


  def __generate_train_validation_sets(self, X_train, Y_train): 
    kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.__RANDOM_SEED)
    new_X_train = []
    new_X_validation = []

    new_Y_train = []
    new_Y_validation = []
    for train_index, test_index in kfold.split(X_train):

      new_X_train.append(X_train[train_index])
      new_Y_train.append(Y_train[train_index])

      new_X_validation.append(X_train[test_index])
      new_Y_validation.append(Y_train[test_index])
      
    return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)

  def search(self, params):
    synthetic_dataset_test_suite = TestSuite()
    features_length = params["features_length"]  
    c_param = params["C"]  
    epsilon_param = params["epsilon"]
    fit_intercept_param = params["fit_intercept"]

    cross_validation_results = []
    for ind_feature in features_length:
      for ind_c in c_param:
        for ind_epsilon in epsilon_param:
          for ind_fit_intercept in fit_intercept_param:

            X_train, _, Y_train, _ = synthetic_dataset_test_suite.__train_and_test_from_numpy_series__(
                transform=min_max_transform,
                series=self.series,
                mixed=True,
                feature_length=ind_feature,
                offset=self.target,
                target_length=1
            )
            Y_train = Y_train.flatten()
            X_trains, X_validations, Y_trains, Y_validations = self.__generate_train_validation_sets(X_train, Y_train)

            validation_loss = np.array([])
            for i in range(len(X_trains)):
              model_linear_regression = Model_Linear_SVR(error_fun=self.loss, C = ind_c, epsilon = ind_epsilon, fit_intercept = ind_fit_intercept)
              model_linear_regression.__train__(X_train=X_trains[i], Y_train=Y_trains[i])

              model_linear_regression.__test__(X_test=X_validations[i], Y_test=Y_validations[i])
              loss = model_linear_regression.__get_error_test__()
              validation_loss = np.append(validation_loss, loss)
            
            mean = np.mean(validation_loss, axis=0)
            std = np.std(validation_loss, axis=0)
            params = {
                "features_length": ind_feature,
                "C": ind_c,
                "epsilon": ind_epsilon,
                "fit_intercept": ind_fit_intercept
            }
            print([mean, std, params, model_linear_regression])
            cross_validation_results.append([mean, std, params, model_linear_regression])
    
    cross_validation_results = np.array(cross_validation_results)
    min = cross_validation_results.T[0].min()
    index_min = np.where(cross_validation_results.T[0] == min)

    return cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]


class KFoldCrossValidation_RBF_SVR():
  __RANDOM_SEED = 1

  def __init__(self, series, loss, one_step_offset_target, kfolds=10):
    self.series = series
    self.loss = loss
    self.target = one_step_offset_target
    self.kfolds = kfolds


  def __generate_train_validation_sets(self, X_train, Y_train): 
    kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.__RANDOM_SEED)
    new_X_train = []
    new_X_validation = []

    new_Y_train = []
    new_Y_validation = []
    for train_index, test_index in kfold.split(X_train):

      new_X_train.append(X_train[train_index])
      new_Y_train.append(Y_train[train_index])

      new_X_validation.append(X_train[test_index])
      new_Y_validation.append(Y_train[test_index])
      
    return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)

  def search(self, params):
    synthetic_dataset_test_suite = TestSuite()
    features_length = params["features_length"]  
    c_param = params["C"]  
    epsilon_param = params["epsilon"]
    gamma_param = params["gamma"]

    cross_validation_results = []
    for ind_feature in features_length:
      for ind_c in c_param:
        for ind_epsilon in epsilon_param:
          for ind_gamma in gamma_param:

            X_train, _, Y_train, _ = synthetic_dataset_test_suite.__train_and_test_from_numpy_series__(
                transform=min_max_transform,
                series=self.series,
                mixed=True,
                feature_length=ind_feature,
                offset=self.target,
                target_length=1
            )
            Y_train = Y_train.flatten()
            X_trains, X_validations, Y_trains, Y_validations = self.__generate_train_validation_sets(X_train, Y_train)

            validation_loss = np.array([])
            for i in range(len(X_trains)):
              model_linear_regression = Model_RBF_SVR(error_fun=self.loss, C = ind_c, epsilon = ind_epsilon, gamma = ind_gamma)
              model_linear_regression.__train__(X_train=X_trains[i], Y_train=Y_trains[i])

              model_linear_regression.__test__(X_test=X_validations[i], Y_test=Y_validations[i])
              loss = model_linear_regression.__get_error_test__()
              validation_loss = np.append(validation_loss, loss)
            
            mean = np.mean(validation_loss, axis=0)
            std = np.std(validation_loss, axis=0)
            params = {
                "features_length": ind_feature,
                "C": ind_c,
                "epsilon": ind_epsilon,
                "gamma": ind_gamma
            }
            print([mean, std, params, model_linear_regression])
            cross_validation_results.append([mean, std, params, model_linear_regression])
    
    cross_validation_results = np.array(cross_validation_results)
    min = cross_validation_results.T[0].min()
    index_min = np.where(cross_validation_results.T[0] == min)

    return cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]  