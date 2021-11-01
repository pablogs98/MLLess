import random


class KeyGenerator:
    def __init__(self, dataset_key, n_workers, n_minibatches, seed=8):
        self.dataset_key = dataset_key
        self.n_workers = n_workers
        self.i = 0
        self.n_minibatches = n_minibatches
        self.minibatches = [n for n in range(n_minibatches)]
        self.rand = random.Random(seed)  # Used for pseudo-random minibatch shuffling

    def get_keys(self, n_keys=1):
        keys = []
        for i in range(n_keys):
            minibatch_id = self.minibatches[self.i]
            object_key = "{}-part{}.pickle".format(self.dataset_key, minibatch_id)
            keys.append(object_key)

            self.prev_i = self.i
            self.i = (self.i + 1) % self.n_minibatches

            # Epoch is over
            if self.i < self.prev_i:
                self.rand.shuffle(self.minibatches)
        return keys
