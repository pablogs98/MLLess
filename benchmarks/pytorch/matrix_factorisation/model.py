import torch
import numpy as np


class MatrixFactorisationModel(torch.nn.Module):
    def __init__(self, num_users, num_items, num_factors, seed=8):
        super().__init__()
        np.random.seed(seed)
        self.L = torch.nn.Embedding(num_users, num_factors)
        self.R = torch.nn.Embedding(num_items, num_factors)

        L_np = np.random.normal(.0, .1, (num_users, num_factors)).astype(np.float32)
        R_np = np.random.normal(.0, .1, (num_items, num_factors)).astype(np.float32)

        self.L.weight.data.copy_(torch.from_numpy(L_np))
        self.R.weight.data.copy_(torch.from_numpy(R_np))

        self.L_bias = torch.nn.Embedding(num_users, 1)
        self.R_bias = torch.nn.Embedding(num_items, 1)

        L_bias_np = np.random.normal(.0, .1, (num_users, 1)).astype(np.float)
        R_bias_np = np.random.normal(.0, .1, (num_items, 1)).astype(np.float)

        self.L_bias.weight.data.copy_(torch.from_numpy(L_bias_np))
        self.R_bias.weight.data.copy_(torch.from_numpy(R_bias_np))

    def forward(self, minibatch):
        users, items = minibatch[:, 0], minibatch[:, 1]
        us, it = self.L(users), self.R(items)
        pred = (us * it).sum(1)
        dpb = pred + self.L_bias(users).squeeze() + self.R_bias(items).squeeze()
        return dpb
