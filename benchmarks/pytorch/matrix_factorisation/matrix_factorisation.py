import torch
from torch.nn import MSELoss
from torch.optim import SGD

from benchmarks.pytorch.matrix_factorisation.model import MatrixFactorisationModel
from benchmarks.pytorch.pytorch_execution import PytorchExecution


class MatrixFactorisation(PytorchExecution):
    def __init__(self, n_users, n_items, n_features, threshold, epochs, execution_name, learning_rate, num_minibatches,
                 dataset_name, dataset_remote_bucket=None, dataset_local_dir=None, seed=8, local=False,
                 write_results=True, adaptive_lr=False, max_time=None, n_threads=1, n_iters=None, sample=False, gpu=False):
        torch.manual_seed(seed)
        model = MatrixFactorisationModel(n_users, n_items, n_features)
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.8)
        loss_func = MSELoss()

        super().__init__(model, optimizer, loss_func, threshold, epochs, execution_name, learning_rate, num_minibatches,
                         dataset_name, dataset_remote_bucket, dataset_local_dir, seed, local, write_results, adaptive_lr,
                         max_time, n_threads, n_iters, sample, gpu)

    def get_samples_labels(self, minibatch):
        data = torch.from_numpy(minibatch)
        rows_cols = data[:, :2].long()
        ratings = data[:, 2].float()
        return rows_cols, ratings
