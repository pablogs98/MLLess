import torch
from torch.nn import BCELoss
from torch.optim import Adam

from benchmarks.pytorch.logistic_regression.model import LogisticRegressionModel
from benchmarks.pytorch.pytorch_execution import PytorchExecution


class LogisticRegression(PytorchExecution):
    def __init__(self, n_features, reg_param, threshold, epochs, execution_name, learning_rate, num_minibatches,
                 dataset_name, bias=True, dataset_remote_bucket=None, seed=8, write_results=True, model=None,
                 sample=False, max_time=None):
        torch.manual_seed(seed)
        self.n_features = n_features

        if model is None:
            model = LogisticRegressionModel(n_features, seed, bias)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=reg_param)
        loss_func = BCELoss()
        print(sample)
        super().__init__(model, optimizer, loss_func, threshold, epochs, execution_name, learning_rate, num_minibatches,
                         dataset_name, dataset_remote_bucket, seed, write_results, sample,
                         max_time=max_time)

    def get_samples_labels(self, minibatch):
        samples_numpy = minibatch[:, 0:minibatch.shape[1] - 1]
        labels_numpy = minibatch[:, minibatch.shape[1] - 1]
        samples = torch.from_numpy(samples_numpy).float()
        labels = torch.from_numpy(labels_numpy).float()
        return samples, labels


class SparseLogisticRegression(LogisticRegression):
    def get_samples_labels(self, minibatch):
        samples_numpy = minibatch[0]
        labels_numpy = minibatch[1]
        labels = torch.from_numpy(labels_numpy).float()
        samples = torch.zeros(len(minibatch[0]), self.n_features)
        for sample in samples_numpy:
            sample = sample[0]
            idx = sample[0]
            value = sample[1]
            samples[int(idx)] = value
        return samples, labels
