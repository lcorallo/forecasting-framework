from calendar import EPOCH
from models.IModel import IModel
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

class CNNSemilinearPredictor(nn.Module):
    def __init__(self, n_inp, layers_fc = 2, dropout=0.2, linear = 180, conv1_out = 6, conv1_kernel = 36, conv2_kernel = 12, n_out = 1):
        super(CNNSemilinearPredictor, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = conv1_out,
            kernel_size = conv1_kernel,
            padding = conv1_kernel - 1
        )
        self.conv2 = nn.Conv1d(
            in_channels = conv1_out,
            out_channels = conv1_out * 2,
            kernel_size = conv2_kernel,
            padding = conv2_kernel - 1
        )
        feature_tensor = self.fe_stack(torch.Tensor([[0]*n_inp]))
        self.semilinear_layers = []
        self.semilinear_layers.append(nn.Linear(feature_tensor.size()[1], linear))
        for i in range(layers_fc - 2):
            self.semilinear_layers.append(nn.Linear(linear, linear))        
        self.semilinear_layers.append(nn.Linear(linear, n_out))
        self.dropout = dropout
        
    def fe_stack(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        return x

    def dm_stack(self, x, train = False):
        y = x
        for i in range(len(self.semilinear_layers)):
            y = F.relu(self.semilinear_layers[i](y))
            if(train is True and i  > 0 and i % 2 == 0):
                y = F.dropout(y, p=self.dropout)
        return y

    def forward(self, x, train=False):
        x = self.fe_stack(x)
        y = self.dm_stack(x, train)
        return y


#TODO: One Step Forecasting               V
#TODO: One Step Forecasting With Offset   V
#TODO: One Shot Forecasting               V
#TODO: One Shot Forecasting With Offset   V
#TODO: Multi Step Forecasting             X
#TODO: Multi Step Forecasting With Offset X
class Model_ConvolutionalSemilinearNN():
    error_train = None;
    error_test = None;
    model = None;
    error_fun = None;

    epochs = None;
    MAX_EPOCHS = 900
    EARLY_STOP_DIFF = 0.005

    def __init__(self, error_fun, semilinear_layers = 2, dropout=0.2, neurons = 180):
        self.error_fun = error_fun
        self.semilinear_layers = semilinear_layers
        self.dropout = dropout
        self.neurons = neurons

    def __get_error_train__(self):
        if(self.error_train is None):
            raise Exception('Sorry, train error train isn\'t set yet')
        return self.error_train

    def __get_error_test__(self):
        if(self.error_test is None):
            raise Exception('Sorry, test error train isn\'t set yet')
        return self.error_test

    def __get_model__(self):
        return self.model

    def __get_yhat_train__(self):
        return self.yhat_train

    def __get_yhat_test__(self):
        return self.yhat_test

    def __test__(self, X_test, Y_test):
        self.model.eval()
        with torch.no_grad():
            self.yhat_test = self.model(X_test)
            test_loss = self.error_fun(Y_test, self.yhat_test)
            return test_loss.item()


    def __train__(self, X_train, Y_train, X_test, Y_test):
        self.model = CNNSemilinearPredictor(
            X_train.size()[1], n_out=Y_train.size()[1],
            layers_fc=self.semilinear_layers,
            dropout=self.dropout,
            linear=self.neurons)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.error_train = []
        self.error_test = []

        epoch_indx = 0
        early_stop = False
        while (epoch_indx < self.MAX_EPOCHS and early_stop is False):
            self.yhat_train = self.model(X_train, train=True)
            train_loss = self.error_fun(Y_train, self.yhat_train)
            self.error_train.append(train_loss.item())

            self.error_test.append(self.__test__(X_test=X_test, Y_test=Y_test))
            if(epoch_indx % 50 == 0 and
                epoch_indx != 0 and
                abs(self.error_train[-50] - self.error_train[-1]) <= self.EARLY_STOP_DIFF):
                early_stop = True

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_indx += 1
        
        self.epochs = self.error_train.index(min(self.error_train))

        if(self.error_train[self.epochs] != self.error_train[-1]): #is not the last value of the array
            self.model = CNNSemilinearPredictor(X_train.size()[1])
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.error_train = []
            self.error_test = []

            for i in range(self.epochs):
                self.yhat_train = self.model(X_train)
                loss = self.error_fun(Y_train, self.yhat_train)
                self.error_train.append(loss.item())

                self.error_test.append(self.__test__(X_test=X_test, Y_test=Y_test))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()