from benchmarks.pywren.logistic_regression.model import LogisticRegression
from benchmarks.pywren.run import run_pywren

n_workers = 24
epochs = 1
# --------------------------------
dataset = 'criteo/criteo-6250'
n_minibatches = 7334
num_features = 13
biased = True
learning_rate = 0.00075
reg_param = 0.05
beta_1 = 0.9
beta_2 = 0.999
# --------------------------------

results_file = f'pywren_criteo-6250-{n_workers}_workers.csv'
models = [LogisticRegression(n_workers=n_workers, biased=biased, num_features=num_features,
                             learning_rate=learning_rate, reg_param=reg_param, seed=8, beta_1=beta_1, beta_2=beta_2,
                             worker_id=worker_id) for worker_id in range(n_workers)]
run_pywren(models, 'bucket', dataset, n_minibatches, epochs, results_file)
