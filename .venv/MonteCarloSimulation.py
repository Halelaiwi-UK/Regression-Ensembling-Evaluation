class MonteCarloSimulation:
    def __init__(self, iter=100, n_models_to_test = 50, rho_range=(0, 1), n_features = 15, train_test_split=0.2):
        self.iterations = iter
        self.n_models = n_models_to_test
        self.rho_range = rho_range
        self.n_features = n_features