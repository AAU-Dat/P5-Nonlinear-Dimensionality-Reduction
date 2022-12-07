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
components_linspace = np.arange(5, 51)
c_logspace = np.logspace(-3, 0, 4)
gamma_logspace = np.logspace(-3, 0, 4)

svm_hyperparameters = [
    {
        "classifier__C": c_logspace
    }
]

pca_hyperparameters = {
    "pca__n_components": components_linspace,
}

lda_hyperparameters = {
    "lda__n_components": components_linspace,
}

isomap_hyperparameters = {
    "isomap__n_components": components_linspace,
}

kernel_pca_hyperparameters = {
    "kernel_pca__n_components": components_linspace,
}


def pca_svm_hyperparameters_function(pca_part=pca_hyperparameters, svm_part=svm_hyperparameters):
    pca_svm_hyperparameters = []

    for hyperparameter in svm_part:
        pca_svm_hyperparameters.append(
            {**pca_part, **hyperparameter})

    return pca_svm_hyperparameters


def lda_svm_hyperparameters_function(lda_part=lda_hyperparameters, svm_part=svm_hyperparameters):
    lda_svm_hyperparameters = []

    for hyperparameter in svm_part:
        lda_svm_hyperparameters.append(
            {**lda_part, **hyperparameter})

    return lda_svm_hyperparameters


def isomap_svm_hyperparameters_function(isomap_part=isomap_hyperparameters, svm_part=svm_hyperparameters):
    isomap_svm_hyperparameters = []

    for hyperparameter in svm_part:
        isomap_svm_hyperparameters.append(
            {**isomap_part, **hyperparameter})

    return isomap_svm_hyperparameters


def kernel_pca_svm_hyperparameters_function(kernel_pca_part=kernel_pca_hyperparameters, svm_part=svm_hyperparameters):
    kernel_pca_svm_hyperparameters = []

    for hyperparameter in svm_part:
        kernel_pca_svm_hyperparameters.append(
            {**kernel_pca_part, **hyperparameter})

    return kernel_pca_svm_hyperparameters


pca_svm_hyperparameters = pca_svm_hyperparameters_function()
lda_svm_hyperparameters = lda_svm_hyperparameters_function()
isomap_svm_hyperparameters = isomap_svm_hyperparameters_function()
kernel_pca_svm_hyperparameters = kernel_pca_svm_hyperparameters_function()
