"""XGB tunig configurations
    """

from scipy.stats import uniform, beta, poisson
GRIDSEARCH_CONFIG = {
    'cv': 5,
    'param_grid': {
        'learning_rate': [0.02],
        'n_estimators': [600],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5],
        'subsample': [0.6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
}


RANDOMIZED_SEARCH_CONFIG = {
    'random_state': 1001,
    'cv': 5,
    'n_iter': 100,
    'param_distributions': {
        'learning_rate': beta(2, 8),
        'n_estimators': [200, 400, 600, 800, 1000],
        'min_child_weight': [1, 5, 10],
        # 'gamma': uniform(0, 10),
        'subsample': [0.6, 0.7, 0.8],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': poisson(3)
    }
}


XGB_STANDARD_GONFIG = {
    'objective': 'binary:logistic',
    'verbosity': 0,
    'nthread': -1
}
