import sys

from benchmarks.pytorch.logistic_regression import SparseLogisticRegression
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

    repetitions = 1
    n_features = 13 + 10 ** 6
    reg_param = 0.05
    threshold = 0.58
    biased = True
    epochs = 100
    learning_rate = 0.00075
    num_minibatches = 7334
    dataset_name = "criteo_sparse/criteo-6250"
    dataset_remote_bucket = "bucket"

    execution_name = f"criteo_sparse-lr-{n_workers}-workers"
    execution = SparseLogisticRegression(n_features, reg_param, threshold, epochs, execution_name, learning_rate,
                                         num_minibatches, dataset_name, biased, dataset_remote_bucket,
                                         max_time=500, sample=True)
    run(n_workers, execution, init_rank, end_rank, address, port)
