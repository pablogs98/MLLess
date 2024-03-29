import sys

from benchmarks.pytorch.matrix_factorisation import MatrixFactorisation
from benchmarks.pytorch.pytorch_execution import run

if __name__ == '__main__':
    n_workers = int(sys.argv[1])
    if len(sys.argv) > 2:  # settings for clusters
        init_rank = int(sys.argv[2])
        end_rank = int(sys.argv[3])
        address = sys.argv[4]
        port = sys.argv[5]
    else:  # default settings
        init_rank = 0
        end_rank = n_workers
        address = '127.0.0.1'
        port = '29500'

    n_users = [69878]
    n_items = [10681]
    num_minibatches = [1600]
    dataset_name = ['movielens/movielens-10m-6250']
    threshold = [0.74]

    n_features = 20
    epochs = 20
    repetitions = 1
    learning_rate = 8.0

    dataset_remote_bucket = "pablo-data"

    for j in range(len(n_users)):
        execution_name = f"{dataset_name[j].replace('/', '')}-mf-{n_workers}-workers"
        execution = MatrixFactorisation(n_users[j], n_items[j], n_features, threshold[j], epochs, execution_name,
                                        learning_rate, num_minibatches[j], dataset_name[j], dataset_remote_bucket)
        run(n_workers, execution, init_rank, end_rank, address, port)
