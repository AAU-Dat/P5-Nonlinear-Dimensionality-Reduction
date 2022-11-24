# This way of declaring the hyperparameters is better for performance. It checks the same things as pca_hyperparameters, but because gamme is only relevant for the rbf kernel, it is not checked for the linear kernel. 5460 vs 10140 runs in this case.
# better_hyperparameters_writing = [
#     {"pca__n_components": [2, 10, 25, 50, 100, 0.95], "classifier__estimator__C": np.logspace(
#         -2, 10, 13), "classifier__estimator__kernel": ["linear"]},
#     {"pca__n_components": [2, 10, 25, 50, 100, 0.95], "classifier__estimator__C": np.logspace(
#         -2, 10, 13), "classifier__estimator__kernel": ["rbf"], "classifier__estimator__gamma": np.logspace(-9, 3, 13)},
# ]

# OG hyperparameters
# pca_hyperparameters = {"pca__n_components": [2, 50, 0.95],
#                        "classifier__estimator__C": [0.01, 0.1, 1],
#                        "classifier__estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
#                        "classifier__estimator__kernel": ["linear"]}

pca_hyperparameters = {
    "pca__n_components": [2, 10, 25, 50, 100, 0.95],
    "classifier__estimator__C": np.logspace(-2, 3, 6),
    "classifier__estimator__gamma": np.logspace(-4, 1, 6),
    "classifier__estimator__kernel": ["rbf", "linear"]
}

lda_hyperparameters = {
    "lda__n_components": [7, 8, 9],
    "lda__store_covariance": [True, False],
    "classifier__estimator__C": np.logspace(-2, 3, 6),
    "classifier__estimator__gamma": np.logspace(-4, 1, 6),
    "classifier__estimator__kernel": ["rbf", "linear"]
}

isomap_hyperparameters = {
    "isomap__n_components": [5, 10, 20, 30, 40],
    "isomap__n_neighbors": [50, 75, 100],
    "classifier__estimator__C": np.logspace(-2, 3, 6),
    "classifier__estimator__gamma": np.logspace(-4, 1, 6),
    "classifier__estimator__kernel": ["rbf", "linear"]
}

kernel_pca_hyperparameters = {
    "kernel_pca__gamma": np.linspace(0.03, 0.05, 10),
    "kernel_pca__kernel": ["rbf", "linear", "poly"],
    "classifier__estimator__C": np.logspace(-2, 3, 6),
    "classifier__estimator__gamma": np.logspace(-4, 1, 6),
    "classifier__estimator__kernel": ["rbf", "linear"]
}
