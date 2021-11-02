from MLLess.utils.prepare_criteo_sparse import load_csv_sparse_tab, dataset_minmax, normalize_dataset
from MLLess.utils.preprocessing import DatasetTypes
from MLLess.utils.preprocessing import Preprocessing

# This is extremely slow and memory-consuming.
# For a quick execution of MLLess, we recommend partitioning Movielens and running PMF.
dataset_path = 'criteo/criteo_train.txt'

with open(dataset_path, 'rb') as file:
    num_lines = Preprocessing._line_count(file)

load_csv_sparse_tab(dataset_path)

# normalize
minmax = dataset_minmax()
normalize_dataset(minmax)

dataset_type = DatasetTypes.CRITEO
dataset_path = 'criteo/sparse_criteo_scale.csv'
minibatch_size = 6250
dataset = 'criteo-sparse-{}'.format(minibatch_size)
separator = ','

p = Preprocessing(dataset_path, dataset_type, separator, n_features=13 + 10 ** 5, sparse=True)
n_minibatches = p.partition(minibatch_size, dataset, scale=False)
print(n_minibatches)
