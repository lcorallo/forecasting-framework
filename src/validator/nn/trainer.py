import torch
from torch import optim
import numpy as np

from src.models.nn.cnn import CNNSemilinearPredictor
from src.models.nn.lstm import LSTMPredictor
from src.models.nn.mlp import MultiLayerPerceptronPredictor
from src.performer.supervised_learning_performer import SlidingWindowPerformer
from src.performer.train_test_performer import TrainTestPerformer
from src.util.parameters import NeuralNetworkTrainingParameters, ModelsIperParameters
from src.util.evaluation import ForecastErrorEvaluation

class Trainer_NeuralNetwork():

    def __init__(self, training_parameters: NeuralNetworkTrainingParameters = NeuralNetworkTrainingParameters):
        self.training_parameters = training_parameters
      
    def _excute_early_stop_training(self, model, x, y, error_performer: ForecastErrorEvaluation):
        optimizer = optim.Adam(model.parameters(), lr=self.training_parameters.LEARNING_RATE)
        epochs_index = 0
        early_stop_condition = False
        loss_distribution = np.array([])
        while (epochs_index < self.training_parameters.EPOCHS and early_stop_condition is False):
            yhat = model(x)
            loss = error_performer.get(y, yhat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_distribution = np.append(loss_distribution, loss.item())            
            if(epochs_index > 400 and epochs_index % 50 == 0 and epochs_index != 0):
                early_stop_condition = bool(np.abs(loss_distribution[-50] - loss_distribution[-1]) <= self.training_parameters.EARLY_STOP_DIFF)
            epochs_index += 1
        return model, loss_distribution

    def _excute_classic_training(self, model, x, y, error_performer: ForecastErrorEvaluation):
        optimizer = optim.Adam(model.parameters(), lr=self.training_parameters.LEARNING_RATE)

        loss_distribution = np.array([])
        for epochs_index in range(self.training_parameters.EPOCHS):
            yhat = model(x)
            loss =  error_performer.get(y, yhat)
            loss_distribution = np.append(loss_distribution, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochs_index += 1
            
        return model, loss_distribution


    def __train__(self, model, x, y, error_performer: ForecastErrorEvaluation):
        if(self.training_parameters.EARLY_STOP):
            model, loss = self._excute_early_stop_training(model, x, y, error_performer = error_performer)
        else:
            model, loss = self._excute_classic_training(model, x, y, error_performer = error_performer)
        return model, loss


class GridSearch_ConvolutionalNeuralNetwork():

    _grid_search_result = []

    def get_all_params(self):
        self._grid_search_result = np.array(self._grid_search_result)
        return self._grid_search_result

    def get_best_params(self):
        results = np.array(self._grid_search_result)
        min = results.T[0].min()
        index_min = np.where(results.T[0] == min)
        return results[index_min][0][0], results[index_min][0][1], results[index_min][0][2]

    def __init__(self, series, target_length=1, target_offset=1, training_parameters = NeuralNetworkTrainingParameters()):
        self.series = series
        self.target_length = target_length
        self.target_offset = target_offset
        self.training_parameters = training_parameters

    
    def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
        features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)

        for ind_feature in features_length:
            SWP = SlidingWindowPerformer(
                feature_length=ind_feature,
                target_length=self.target_length,
                target_offset=self.target_offset
            )

            TTP = TrainTestPerformer(portion_train = 0.8, random_sampling = True)
            _, x, y = SWP.get(self.series)

            X_train, X_test, Y_train, Y_test = TTP.get(x, y)

            X_train = torch.tensor(data = X_train, dtype = torch.float32)
            Y_train = torch.tensor(data = Y_train, dtype = torch.float32)
            X_test = torch.tensor(data = X_test, dtype = torch.float32)
            Y_test = torch.tensor(data = Y_test, dtype = torch.float32)

            model = CNNSemilinearPredictor(
                n_inp = ind_feature,
                n_out = self.target_length,
                conv1_out = 6,
                conv1_kernel = 36,
                conv2_kernel = 12,
                layers_fc = 2,
                linear = 180
            )
            
            trainer = Trainer_NeuralNetwork(training_parameters = self.training_parameters)
            model, _ = trainer.__train__(model = model, x = X_train, y = Y_train, error_performer = error_performer)

            model.eval()
            with torch.no_grad():
                yhat = model(X_test)
                loss = error_performer.get(Y_test, yhat)

            self._grid_search_result.append([
                loss.item(),
                {
                    ModelsIperParameters.FEATURE_LENGTH: ind_feature
                },
                model
            ])
            print(loss.item(), ind_feature)

class GridSearch_LongShortTermNeuralNetwork():

    _grid_search_result = []

    def get_all_params(self):
        self._grid_search_result = np.array(self._grid_search_result)
        return self._grid_search_result

    def get_best_params(self):
        results = np.array(self._grid_search_result)
        min = results.T[0].min()
        index_min = np.where(results.T[0] == min)
        return results[index_min][0][0], results[index_min][0][1], results[index_min][0][2]
        
    def __init__(self, series, target_length=1, target_offset=1, training_parameters = NeuralNetworkTrainingParameters()):
        self.series = series
        self.target_length = target_length
        self.target_offset = target_offset
        self.training_parameters = training_parameters

    
    def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
        features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)

        for ind_feature in features_length:
            SWP = SlidingWindowPerformer(
                feature_length=ind_feature,
                target_length=self.target_length,
                target_offset=self.target_offset
            )

            TTP = TrainTestPerformer(portion_train = 0.8, random_sampling = True)
            _, x, y = SWP.get(self.series)

            X_train, X_test, Y_train, Y_test = TTP.get(x, y)

            X_train = torch.tensor(data = X_train, dtype = torch.float32)
            Y_train = torch.tensor(data = Y_train, dtype = torch.float32)
            X_test = torch.tensor(data = X_test, dtype = torch.float32)
            Y_test = torch.tensor(data = Y_test, dtype = torch.float32)

            model = LSTMPredictor(
                input_dim = ind_feature,
                output_dim = self.target_length,
                num_layers = 2,
                hidden = 128,
                dropout = 0.2
            )
            trainer = Trainer_NeuralNetwork(training_parameters = self.training_parameters)

            model, _ = trainer.__train__(model = model, x = X_train, y = Y_train, error_performer = error_performer)
            model.eval()
            with torch.no_grad():
                yhat = model(X_test)
                loss = error_performer.get(Y_test, yhat)

            self._grid_search_result.append([
                loss.item(),
                {
                    ModelsIperParameters.FEATURE_LENGTH: ind_feature
                },
                model
            ])
            print(loss.item(), ind_feature)


class GridSearch_MultiLayerPerceptronNetwork():

    _grid_search_result = []

    def get_all_params(self):
        self._grid_search_result = np.array(self._grid_search_result)
        return self._grid_search_result

    def get_best_params(self):
        results = np.array(self._grid_search_result)
        min = results.T[0].min()
        index_min = np.where(results.T[0] == min)
        return results[index_min][0][0], results[index_min][0][1], results[index_min][0][2]
        
    def __init__(self, series, target_length=1, target_offset=1, training_parameters = NeuralNetworkTrainingParameters()):
        self.series = series
        self.target_length = target_length
        self.target_offset = target_offset
        self.training_parameters = training_parameters

    
    def search(self, params: ModelsIperParameters, error_performer: ForecastErrorEvaluation):
        features_length = params.get(ModelsIperParameters.FEATURE_LENGTH)

        for ind_feature in features_length:
            SWP = SlidingWindowPerformer(
                feature_length=ind_feature,
                target_length=self.target_length,
                target_offset=self.target_offset
            )

            TTP = TrainTestPerformer(portion_train = 0.7, random_sampling = True)
            _, x, y = SWP.get(self.series)

            X_train, X_test, Y_train, Y_test = TTP.get(x, y)

            X_train = torch.tensor(data = X_train, dtype = torch.float32)
            Y_train = torch.tensor(data = Y_train, dtype = torch.float32)
            X_test = torch.tensor(data = X_test, dtype = torch.float32)
            Y_test = torch.tensor(data = Y_test, dtype = torch.float32)

            model = MultiLayerPerceptronPredictor(input_dim= ind_feature, output_dim= self.target_length)
            trainer = Trainer_NeuralNetwork(training_parameters = self.training_parameters)

            model, _ = trainer.__train__(model = model, x = X_train, y = Y_train, error_performer = error_performer)
            model.eval()
            with torch.no_grad():
                yhat = model(X_test)
                loss = error_performer.get(Y_test, yhat)

            self._grid_search_result.append([
                loss.item(),
                {
                    ModelsIperParameters.FEATURE_LENGTH: ind_feature
                },
                model
            ])
            # print(loss.item(), ind_feature)