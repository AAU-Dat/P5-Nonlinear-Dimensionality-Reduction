import idx2numpy

from library import pca_results
from load_mnist import load_mnist


def main():
    load_mnist()
    pca_hyperparameters = {
        "pca__n_components": [2, 50, 100, 0.95],
        "pca__whiten": [True, False],
        "classifier__estimator__C": [0.01, 0.1, 1, 10],
        "classifier__estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    }
    X = idx2numpy.convert_from_file(
        'src/mnist_data/train_file_image').reshape(60000, 784)
    y = idx2numpy.convert_from_file('src/mnist_data/train_file_label')
    X_test = idx2numpy.convert_from_file(
        'src/mnist_data/test_file_image').reshape(10000, 784)
    y_test = idx2numpy.convert_from_file('src/mnist_data/test_file_label')

    # Only use 400 samples to speed up the process. Change to full size for final run
    pca_results(X[:400], y[:400], X_test, y_test, pca_hyperparameters)


#     # load the data
#     loading_time_start = time.time()
#     (training_data, test_data) = load_mnist.load_mnist()
#     # remove pixels from the data
#     loading_time_end = time.time()
#     print("Loading time: " + str(loading_time_end - loading_time_start))

#     pca_time_start = time.time()
#     (training_data2, test_data2) = pca.pca((training_data, test_data), 1)
#     pca_time_end = time.time()

#     print("PCA time: " + str(pca_time_end - pca_time_start))
#     (training_data, _) = load_mnist.load_mnist()
#     # create the model

#     training_time_start = time.time()
#     model = svm_model.train_model(training_data, svm.SVC())
#     training_time_end_first = time.time()
#     model2 = svm_model.train_model(training_data2, svm.SVC())
#     training_time_end_second = time.time()

#     print("Training time first: " +
#           str(training_time_end_first - training_time_start))
#     print("Training time second: " +
#           str(training_time_end_second - training_time_end_first))
#     # create the confusion matrix
#     print(confusion_matrix.create_confusion_matrix(model, test_data))
#     print(confusion_matrix.create_confusion_matrix(model2, test_data2))


if __name__ == "__main__":
    main()
