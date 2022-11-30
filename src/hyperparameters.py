import numpy as np

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

# concatenate two lists

c_logspace = np.logspace(0, 4, 5)
gamma_logspace = np.logspace(-4, 0, 5)

svm_hyperparameters = [
    {
        "classifier__estimator__kernel": ["linear"],
        "classifier__estimator__C": c_logspace
    },
    {
        "classifier__estimator__kernel": ["rbf"],
        "classifier__estimator__C": c_logspace,
        "classifier__estimator__gamma": gamma_logspace
    }
]

pca_hyperparameters = {
    "pca__n_components": [10, 25, 50, 100, 0.95],
    "pca__whiten": [True, False],
}

lda_hyperparameters = {
    "lda__n_components": [7, 8, 9],
    "lda__store_covariance": [True, False]
}

isomap_hyperparameters = {
    "isomap__n_components": [2, 3, 5, 10, 25, 50, 100],
    "isomap__n_neighbors": [None],
    "isomap__n_radius": [7.5, 8, 8.5, 9, 9.5, 10]
}

kernel_pca_hyperparameters = {
    "kernel_pca__gamma": np.linspace(0.03, 0.05, 10),
    "kernel_pca__kernel": ["rbf", "sigmoid", "poly"]
}

pca_svm_hyperparameters = []
lda_svm_hyperparameters = []
isomap_svm_hyperparameters = []
kernel_pca_svm_hyperparameters = []

for hyperparameter in svm_hyperparameters:
    pca_svm_hyperparameters.append({**pca_hyperparameters, **hyperparameter})
    lda_svm_hyperparameters.append({**lda_hyperparameters, **hyperparameter})
    isomap_svm_hyperparameters.append(
        {**isomap_hyperparameters, **hyperparameter})
    kernel_pca_svm_hyperparameters.append(
        {**kernel_pca_hyperparameters, **hyperparameter})
