from MLLess.algorithms import ProbabilisticMatrixFactorisation

# Dataset specific parameters
dataset = 'movielens/movielens-20m-12000'
n_minibatches = 1666
n_users = 138493
n_items = 27278

# Create and run PMF job (baseline, no optimisations)
model = ProbabilisticMatrixFactorisation(n_workers=24, n_users=n_users, n_items=n_items, n_factors=20,
                                         learning_rate=8.0, end_threshold=0.68)
model.fit(dataset, n_minibatches)
model.generate_stats('movielens-20m_results')
