from MLLess.algorithms import ProbabilisticMatrixFactorisation

# Dataset specific parameters
dataset = 'movielens/movielens-20m-12000'
n_minibatches = 1666
n_users = 138493
n_items = 27278

# End criteria
end_threshold = 0.68  # Stop when a loss of 0.68 MSE is reached...
max_time = 500  # ... stop when 500s have elapsed ...
max_epochs = 10  # ... or 10 epochs have been completed.

# Create and run PMF job with the end criteria above
model = ProbabilisticMatrixFactorisation(n_workers=24,
                                         n_users=n_users, n_items=n_items, n_factors=20, learning_rate=8.0,
                                         significance_threshold=0.7,
                                         # End criteria:
                                         max_epochs=max_epochs, max_time=max_time, end_threshold=end_threshold)
model.fit(dataset, n_minibatches)
model.generate_stats('movielens-20m_results')
