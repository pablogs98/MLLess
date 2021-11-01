import torch
import numpy as np


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, num_features, seed=8, bias=True):
        super().__init__()
        np.random.seed(seed)
        self.weights = torch.nn.Linear(num_features, 1, bias=bias)
        k = 1.0 / num_features
        np_weights = np.random.uniform(-k, k, num_features)
        self.weights.weight.data.copy_(torch.from_numpy(np_weights))
        if bias:
            np_bias = np.random.uniform(-k, k, 1)
            self.weights.bias.data.copy_(torch.from_numpy(np_bias))

    def forward(self, x):
        return torch.squeeze(torch.sigmoid(self.weights(x)))
