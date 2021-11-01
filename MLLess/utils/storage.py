import pickle
import time
import zlib
import random

from MLLess.utils.storage_backends import CosBackend


class StorageIterator:
    def __init__(self, storage, bucket, dataset, worker_id, n_minibatches, seed, n_workers, dataset_path):
        self.i = worker_id
        self.worker_id = worker_id
        self.pickler = CosBackend(storage, bucket)
        self.rand = random.Random(seed)  # Used for pseudo-random minibatch shuffling
        self.minibatches = list(range(n_minibatches))
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.local = False
        self.n_workers = n_workers
        self.n_minibatches = n_minibatches
        self.bucket = bucket
        self.cos_time = 0

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.time()
        minibatch_id = self.minibatches[self.i]
        object_key = "{}-part{}.pickle".format(self.dataset, minibatch_id)
        if not self.local:
            minibatch = self.pickler.get(object_key)
        else:
            with open('{}/{}'.format(self.dataset_path, object_key), "rb") as f:
                minibatch = zlib.decompress(f.read())
                minibatch = pickle.loads(minibatch)

        self.prev_i = self.i
        self.i = (self.i + self.n_workers) % self.n_minibatches

        # Epoch is over
        if self.i < self.prev_i:
            self.rand.shuffle(self.minibatches)
        t1 = time.time()
        self.cos_time = t1 - t0
        return minibatch
