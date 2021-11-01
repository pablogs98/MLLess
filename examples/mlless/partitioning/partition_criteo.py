from MLLess.utils.preprocessing import DatasetTypes
from MLLess.utils import Preprocessing

# todo: add previous preprocessing

dataset_type = DatasetTypes.CRITEO
dataset_path = 'criteo/sparse_criteo_scale.csv'
minibatch_size = 6250
dataset = 'criteo-sparse-{}'.format(minibatch_size)
separator = ','

p = Preprocessing(dataset_path, dataset_type, separator, n_features=13 + 10 ** 5, sparse=True)
n_minibatches = p.partition(minibatch_size, dataset, scale=False)  # Criteo dataset is already scaled (scale=False)
print(n_minibatches)
