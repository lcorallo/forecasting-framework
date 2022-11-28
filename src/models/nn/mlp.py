import torch
from torch import nn, optim
from .interface import INeuralNetwork

class MultiLayerPerceptronPredictor(INeuralNetwork):

  def __init__(self, input_dim=1, output_dim=1):
    super(MultiLayerPerceptronPredictor, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.sequential = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )

  def forward(self, x):
      return self.sequential(x)

  def __identify__(self):
      return "AR_MultiLayerPerceptronNeuralNetwork("+str(self.input_dim)+");"
