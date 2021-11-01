from benchmarks.pywren.matrix_factorisation.model import MatrixFactorisation
from benchmarks.pywren.run import run_pywren

n_workers = 24
repetitions = 1
epochs = 1
# --------------------------------
learning_rate = 8.0
learning_rate_l = learning_rate
learning_rate_r = learning_rate
n_factors = 20
lambda_l = 0.1
lambda_r = 0.1
momentum = 0.8
dataset = 'movielens/movielens-20m-12000'
n_minibatches = 1666  # mb size = 6250
n_users = 138493
n_items = 27278
# --------------------------------


results_file = f'ml-20m-{n_workers}_workers.csv'
models = [MatrixFactorisation(n_workers=n_workers, learning_rate_r=learning_rate_r, learning_rate_l=learning_rate_l,
                              n_factors=n_factors, lambda_r=lambda_r, lambda_l=lambda_l, momentum=momentum, seed=8,
                              n_items=n_items, n_users=n_users, worker_id=worker_id)
          for worker_id in range(n_workers)]
run_pywren(models, 'bucket', dataset, n_minibatches, epochs, results_file)
