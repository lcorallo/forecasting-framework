import numpy as np
import torch
from .goal import ForecastingGoal

class ForecastErrorEvaluation():
    def RMSE_Loss(actual, predicted):
        """Root Mean Squared Error"""
        if torch.is_tensor(actual):
            return torch.sqrt(torch.mean(torch.square(actual - predicted)))
        return np.sqrt(np.mean(np.square(actual - predicted)))


    def __init__(self, goal:ForecastingGoal, loss=RMSE_Loss):
        self.goal = goal
        self.loss = loss

    def get(self, y, yhat):
        if(self.goal.get() == "ONE_STEP"):
            return self.loss(y, yhat)
        
        if(self.goal.get() == "ONE_SHOT"):
            if torch.is_tensor(y): return (self.loss(y, yhat))
            temp_loss = []
            for i in range(y.shape[1]):
                temp_loss.append(self.loss(y.T[i], yhat.T[i]))
            return np.mean(np.array(temp_loss))
        raise "Not Implemented"
