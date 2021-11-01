from MLLess.algorithms import ProbabilisticMatrixFactorisation

# Dataset specific parameters
dataset = 'movielens/movielens-20m-12000'
n_minibatches = 1666
n_users = 138493
n_items = 27278

# Auto-tuner parameters
remove_interval = 20  # Worker removal interval (in seconds)
remove_threshold = 15  # Auto-tuner loss degradation threshold (%)
min_workers = 10  # A minimum number of workers can be specified, not compulsory

# Create and run PMF job
model = ProbabilisticMatrixFactorisation(n_workers=24,
                                         n_users=n_users, n_items=n_items, n_factors=20,
                                         end_threshold=0.68, significance_threshold=0.7,
                                         learning_rate=8.0,
                                         # To enable the scale-in auto-tuner add, at least, these two parameters:
                                         remove_threshold=remove_threshold, remove_interval=remove_interval,
                                         # Additionally:
                                         min_workers=min_workers)
model.fit(dataset, n_minibatches)
model.generate_stats('movielens-20m_results')
