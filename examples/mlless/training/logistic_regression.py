from MLLess.algorithms.logistic_regression import SparseLogisticRegression

# Dataset specific parameters
dataset = 'criteo_sparse/criteo-6250'
num_minibatches = 7334
num_features = 13 + 10 ** 5
lr = 0.00075
reg_param = 0.05

# Create and run LR job (baseline, no optimisations)
alg = SparseLogisticRegression(n_workers=24, num_features=num_features, reg_param=reg_param, learning_rate=lr,
                               end_threshold=0.65)

result = alg.fit(dataset, num_minibatches)
alg.generate_stats('criteo_results')
