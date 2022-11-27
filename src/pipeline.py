import idx2numpy

from library import pca_results
from load_mnist import load_mnist
from convonet import *
from load_helpers import normalize_data
from cnn_model_lib import *
def main():
    load_mnist()
    pca_hyperparameters = {
        "pca__n_components": [2, 50, 100, 0.95],
        "pca__whiten": [True, False],
        "classifier__estimator__C": [0.01, 0.1, 1, 10],
        "classifier__estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    }
    other_hyperparameters_for_svm = {
        "classifier__estimator__C": [0.01, 0.1, 1, 10],
        "classifier__estimator__gamma": [10, 1, 0.1, 0.01, 0.001, 0.0001],
        "classifier__estimator__kernel": ["rbf", "poly", "sigmoid"], #we might not need linear actually
        "classifier__estimator__max_iter": [1000, 2000, 3000, 4000, 5000],
        "classifier__estimator__degree": [2, 3, 4, 5, 6, 7, 8, 9, 10] #this accounts only for polynomial svm
    }

    X = idx2numpy.convert_from_file(
        'src/mnist_data/train_file_image').reshape(60000, 784)
    y = idx2numpy.convert_from_file('src/mnist_data/train_file_label')
    X_test = idx2numpy.convert_from_file(
        'src/mnist_data/test_file_image').reshape(10000, 784)
    y_test = idx2numpy.convert_from_file('src/mnist_data/test_file_label')

    # Only use 400 samples to speed up the process. Change to full size for final run
    #pca_results(X[:400], y[:400], X_test, y_test, pca_hyperparameters)
    #add hyperparameters for cross-validation for cnn dictionary

    hyperparameters_for_cnn_for_pca = {
        'pca__n_components': [16],
        'classifier__input_shape': [(4, 4, 1)],
        'classifier__batch_size': [128, 256],
        'classifier__init_mode': ['he_uniform', 'lecun_uniform'],
    }

    #PCA: CNN needs square matrices, which is from 2x2 to 28x28. The values in the range can be changed
    hyperparameters_pca_cnn = []

    for value in range(10, 12):
        hyperparameters_pca_cnn.append(
        {"pca__n_components": [value*value],
         "classifier__input_shape": [(value, value, 1)],
         "classifier__batch_size": [128, 256],
         "classifier__init_mode": ["he_uniform", "lecun_uniform", "normal"], 
         "classifier__pool_type": ["max", "avg"]
         }
    )

    #LDA reduces the dimensions to maximum 9. CNN needs to use square matrices, which means that we only use 2x2 and 3x3
    hyperparameters_lda_cnn = [
    {
        'lda__n_components': [4],
        'classifier__input_shape': [(2, 2, 1)],
        'classifier__batch_size': [128, 256],
        'classifier__init_mode': ['he_uniform', 'normal'],
        'classifier__pool_type': ['max', 'avg']
    },
    {
        'lda__n_components': [9],
        'classifier__input_shape': [(3,3, 1)],
        'classifier__batch_size': [128, 256],
        'classifier__init_mode': ['he_uniform', 'normal'],
        'classifier__pool_type': ['max', 'avg']
    }
    ]
    #KernelPCA has has more hyperparameters than PCA, since it is nonlinear
    hyperparameters_kernelpca_cnn = []
    for value in range(4,6):
        hyperparameters_kernelpca_cnn.append(
        {
            'kernelpca__n_components': [value*value],
            'kernelpca__kernel': ['rbf', 'poly', 'sigmoid'],
            'kernelpca__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernelpca__degree': [4,8,12], #relevant only for poly

            'classifier__input_shape': [(value, value, 1)],
            'classifier__batch_size': [128, 256],
            'classifier__init_mode': ['he_uniform', 'normal'],
            'classifier__pool_type': ['max', 'avg']
        }
    )

    #Isomap has number of neighbors that can be adjusted
    hyperparameters_isomap_cnn = []
    for value in range(2,3):
        hyperparameters_isomap_cnn.append({
            'isomap__n_components': [value*value],
            'isomap__n_neighbors': [8, 10, 12],
            'classifier__input_shape': [(value, value, 1)],
            'classifier__batch_size': [128, 256],
            'classifier__init_mode': ['he_uniform', 'normal'],
            'classifier__pool_type': ['max', 'avg']
        }
        )

    hyperparameters_cnn = {
        'classifier__input_shape': [(28, 28, 1)],
        'classifier__batch_size': [256],
        'classifier__init_mode': ['lecun_uniform', 'normal'],
        'classifier__pool_type': ['max', 'avg']
        }
    n_samples = 2352

#    Uncomment those lines to run the CNN with the methods
#    lda_cnn_results(X[:n_samples], y[:n_samples], X_test[:n_samples], y_test[:n_samples], hyperparameters_lda_cnn)
#    pca_cnn_results(X[:n_samples], y[:n_samples], X_test[:n_samples], y_test[:n_samples], hyperparameters_pca_cnn)
#    kernel_pca_results(X[:n_samples], y[:n_samples], X_test[:n_samples], y_test[:n_samples], hyperparameters_kernel_pca_cnn)    
#    isomap_cnn_results(X[:n_samples], y[:n_samples], X_test[:n_samples], y_test[:n_samples], hyperparameters_isomap_cnn)

# this is only for the model
#    cnn_results(X[:n_samples], y[:n_samples], X_test[:n_samples], y_test[:n_samples], hyperparameters_cnn)


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

# def normalize_for_cnn():
#     #the ConvNet needs the data to be in this shape
#     X = idx2numpy.convert_from_file(
#         'src/mnist_data/train_file_image').reshape(60000, 28,28,1)
#     y = idx2numpy.convert_from_file('src/mnist_data/train_file_label')
#     X_test = idx2numpy.convert_from_file(
#         'src/mnist_data/test_file_image').reshape(10000, 28,28,1)
#     y_test = idx2numpy.convert_from_file('src/mnist_data/test_file_label')
    
#     data = ((X,y), (X_test, y_test))
#     (normalized_train, normalized_test) = normalize_data(data)
#     X = normalized_train[0]
#     y = normalized_train[1]
#     X_test = normalized_test[0]
#     y_test = normalized_test[1]
#     return X, y, X_test, y_test

if __name__ == "__main__":
    main()
