import sys
import idx2numpy

from hyperparameters import (isomap_svm_hyperparameters,
                             kernel_pca_svm_hyperparameters,
                             lda_svm_hyperparameters, pca_svm_hyperparameters,
                             svm_hyperparameters)
from library import (baseline_svm_results, isomap_svm_results,
                     kernel_pca_svm_results, lda_svm_results, pca_svm_results)
from load_mnist import load_mnist


def main(datasize=60000, nldr_datasize=15000):
    load_mnist()

    X = idx2numpy.convert_from_file(
        'src/mnist_data/train_file_image').reshape(60000, 784)
    y = idx2numpy.convert_from_file('src/mnist_data/train_file_label')
    X_test = idx2numpy.convert_from_file(
        'src/mnist_data/test_file_image').reshape(10000, 784)
    y_test = idx2numpy.convert_from_file('src/mnist_data/test_file_label')

    # baseline_svm_results(
    #     X[:datasize], y[:datasize], X_test, y_test,
    #     svm_hyperparameters, "baseline_svm_" + str(datasize))

    # pca_svm_results(
    #     X[:datasize], y[:datasize], X_test, y_test,
    #     pca_svm_hyperparameters, "pca_svm_lars_" + str(datasize))

    # lda_svm_results(
    #     X[:datasize], y[:datasize], X_test, y_test,
    #     lda_svm_hyperparameters, "lda_svm_" + str(datasize))

    # isomap_svm_results(
    #     X[:nldr_datasize], y[:nldr_datasize], X_test, y_test,
    #     isomap_svm_hyperparameters, "isomap_svm_" + str(nldr_datasize))

    # kernel_pca_svm_results(
    #     X[:nldr_datasize], y[:nldr_datasize], X_test, y_test,
    #     kernel_pca_svm_hyperparameters, "kernel_pca_svm_" + str(nldr_datasize))

    isomap_svm_results(
        X[:nldr_datasize], y[:nldr_datasize], X_test, y_test,
        isomap_svm_hyperparameters, "isomap_svm_lars_" + str(nldr_datasize))



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(
            int(sys.argv[1]),
            int(sys.argv[1]))
    elif len(sys.argv) > 2:
        main(
            int(sys.argv[1]),
            int(sys.argv[2]))
    else:
        main()
