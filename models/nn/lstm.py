import torch
from torch import nn, optim

class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LSTMPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = 128,
            num_layers = 2,
            dropout = .1
        )
        self.linear = nn.Linear(in_features=128, out_features=output_dim)

    def forward(self, sequences):
        h0, c0 = torch.zeros((2, 128), dtype=torch.float32), torch.zeros((2, 128), dtype=torch.float32)
        y_lstm, _ = self.lstm(sequences, (h0, c0))
        y_pred = self.linear(y_lstm)
        return y_pred



#TODO: One Step Forecasting               V
#TODO: One Step Forecasting With Offset   V
#TODO: One Shot Forecasting               V
#TODO: One Shot Forecasting With Offset   V
#TODO: Multi Step Forecasting             X
#TODO: Multi Step Forecasting With Offset X
class Model_LSTM():
    error_train = None;
    error_test = None;
    model = None;
    error_fun = None;

    epochs = None;
    MAX_EPOCHS = 900
    EARLY_STOP_DIFF = 0.005

    def __init__(self, error_fun):
        self.error_fun = error_fun

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
        self.model = LSTMPredictor(input_dim=X_train.size()[1], output_dim=Y_train.size()[1])
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.error_train = []
        self.error_test = []

        epoch_indx = 0
        early_stop = False
        while (epoch_indx < self.MAX_EPOCHS and early_stop is False):
            self.yhat_train = self.model(X_train)
            train_loss = self.error_fun(Y_train, self.yhat_train)
            self.error_train.append(train_loss.item())

            if(epoch_indx % 50 == 0 and
                epoch_indx != 0):
                print("EPOCH", epoch_indx);
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
            self.model = LSTMPredictor(input_dim=X_train.size()[1], output_dim=Y_train.size()[1])
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