from MLLess.utils import Preprocessing
from MLLess.utils.preprocessing import DatasetTypes

dataset_type = DatasetTypes.MOVIELENS
dataset_path = 'ml-20m/ratings.csv'
minibatch_size = 4000
dataset_name = f'movielens/movielens-20m-{minibatch_size}'
separator = ','

p = Preprocessing(dataset_path, dataset_type, separator)
n_minibatches = p.partition(minibatch_size, dataset_name)
print(n_minibatches)
