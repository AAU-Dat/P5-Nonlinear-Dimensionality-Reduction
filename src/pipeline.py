import idx2numpy

from hyperparameters import (isomap_svm_hyperparameters,
                             kernel_pca_svm_hyperparameters,
                             lda_svm_hyperparameters, pca_svm_hyperparameters)
from library import (isomap_svm_results, kernel_pca_svm_results,
                     lda_svm_results, pca_svm_results)
from load_mnist import load_mnist


def main():
    load_mnist()

    X = idx2numpy.convert_from_file(
        'src/mnist_data/train_file_image').reshape(60000, 784)
    y = idx2numpy.convert_from_file('src/mnist_data/train_file_label')
    X_test = idx2numpy.convert_from_file(
        'src/mnist_data/test_file_image').reshape(10000, 784)
    y_test = idx2numpy.convert_from_file('src/mnist_data/test_file_label')

    pca_svm_results(X, y, X_test, y_test,
                    pca_svm_hyperparameters, "pca_svm")

    lda_svm_results(X, y, X_test, y_test, lda_svm_hyperparameters, "lda_svm")

    isomap_svm_results(
        X, y, X_test, y_test, isomap_svm_hyperparameters, "isomap_svm")

    kernel_pca_svm_results(X, y, X_test, y_test,
                           kernel_pca_svm_hyperparameters, "kernel_pca_svm")


if __name__ == "__main__":
    main()
