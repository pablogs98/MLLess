from MLLess.algorithms import ProbabilisticMatrixFactorisation

# Dataset specific parameters
dataset = 'movielens/movielens-20m-12000'
n_minibatches = 1666
n_users = 138493
n_items = 27278

# Significance filter parameter
significance_threshold = 0.7    # Higher: more strict, less communication. Lower: less strict, more communication.

# Create and run PMF job
model = ProbabilisticMatrixFactorisation(n_workers=24,
                                         n_users=n_users, n_items=n_items, n_factors=20,
                                         end_threshold=0.68, learning_rate=8.0,
                                         # To enable the scale-in auto-tuner add the following parameter:
                                         significance_threshold=significance_threshold)
model.fit(dataset, n_minibatches)
model.generate_stats('movielens-20m_results')
