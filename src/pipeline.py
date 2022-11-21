import time

from sklearn import svm

import confusion_matrix
import load_mnist
import pca
import svm_model


def main():
    # load the data
    loading_time_start = time.time()
    (training_data, test_data) = load_mnist.load_mnist()
    # remove pixels from the data
    loading_time_end = time.time()
    print("Loading time: " + str(loading_time_end - loading_time_start))

    pca_time_start = time.time()
    (training_data2, test_data2) = pca.pca((training_data, test_data), 1)
    pca_time_end = time.time()

    print("PCA time: " + str(pca_time_end - pca_time_start))
    (training_data, _) = load_mnist.load_mnist()
    # create the model

    training_time_start = time.time()
    model = svm_model.train_model(training_data, svm.SVC())
    training_time_end_first = time.time()
    model2 = svm_model.train_model(training_data2, svm.SVC())
    training_time_end_second = time.time()

    print("Training time first: " +
          str(training_time_end_first - training_time_start))
    print("Training time second: " +
          str(training_time_end_second - training_time_end_first))
    # create the confusion matrix
    print(confusion_matrix.create_confusion_matrix(model, test_data))
    print(confusion_matrix.create_confusion_matrix(model2, test_data2))


main()
