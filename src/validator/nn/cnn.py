from sklearn.model_selection import KFold
import numpy as np
import torch

from synthetic.test_suite import TestSuite
from src.models.nn.cnn import Model_ConvolutionalSemilinearNN


class KFoldCrossValidation_ConvolutionalSemilinearNN():
  __RANDOM_SEED = 1

  def __init__(self, series, loss, one_step_offset_target, kfolds=10):
    self.series = series
    self.loss = loss
    self.target = one_step_offset_target
    self.kfolds = kfolds


  def __generate_train_validation_sets(self, X_train, Y_train, tensor=False): 
    kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.__RANDOM_SEED)
    new_X_train = []
    new_X_validation = []

    new_Y_train = []
    new_Y_validation = []
    for train_index, test_index in kfold.split(X_train):

      if(tensor is True):
        new_X_train.append(torch.tensor(X_train[train_index], dtype=torch.float32))
        new_Y_train.append(torch.tensor(Y_train[train_index], dtype=torch.float32))

        new_X_validation.append(torch.tensor(X_train[test_index], dtype=torch.float32))
        new_Y_validation.append(torch.tensor(Y_train[test_index], dtype=torch.float32))
      else:
        new_X_train.append(X_train[train_index])
        new_Y_train.append(Y_train[train_index])

        new_X_validation.append(X_train[test_index])
        new_Y_validation.append(Y_train[test_index])

    return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)

  def search(self, params):
    synthetic_dataset_test_suite = TestSuite()
    features_length = params["features_length"]
    semilinear_layers = params["semilinear_layers"]
    neurons_decision_making = params["neurons_decision_making"]
    dropout = params["dropout"]
    tensor = params["options"]["tensor"]

    cross_validation_results = []
    for ind_feature in features_length:
      for ind_semilinear_layers in semilinear_layers:
        for ind_neurons in neurons_decision_making:
          for ind_dropout in dropout:
              X_train, _, Y_train, _ = synthetic_dataset_test_suite.__train_and_test_from_numpy_series__(
                  transform=min_max_transform,
                  series=self.series,
                  mixed=True,
                  feature_length=ind_feature,
                  offset=self.target,
                  target_length=1
              )

              X_trains, X_validations, Y_trains, Y_validations = self.__generate_train_validation_sets(X_train, Y_train, tensor=tensor)

              validation_loss = np.array([])
              for i in range(len(X_trains)):
                model_nn_cnn = Model_ConvolutionalSemilinearNN(error_fun=self.loss, semilinear_layers=ind_semilinear_layers,neurons=ind_neurons, dropout=ind_dropout)
                model_nn_cnn.__train__(X_train=X_trains[i], Y_train=Y_trains[i], X_test=X_validations[i], Y_test=Y_validations[i])

                loss = model_nn_cnn.__get_error_test__()
                validation_loss = np.append(validation_loss, loss)
              
              mean = np.mean(validation_loss, axis=0)
              std = np.std(validation_loss, axis=0)
              params = {
                "features_length": ind_feature,
                "semilinear_layers": ind_semilinear_layers,
                "neurons_decision_making": ind_neurons,
                "dropout": ind_dropout
              }
              cross_validation_results.append([mean, std, params, model_nn_cnn])
    
    cross_validation_results = np.array(cross_validation_results)
    min = cross_validation_results.T[0].min()
    index_min = np.where(cross_validation_results.T[0] == min)

    return cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]  