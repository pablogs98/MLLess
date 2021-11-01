import lithops

from MLLess.utils import StorageIterator


class Sampler:
    def __init__(self, dataset_remote_bucket, dataset_name, rank, num_minibatches, seed, n_workers):
        iterator = StorageIterator(lithops.storage.Storage().get_client(), dataset_remote_bucket,
                                   dataset_name, rank, num_minibatches, seed, n_workers, None)
        self.iterator = iter(iterator)
        self.minibatch = None

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        return next(self.iterator)
