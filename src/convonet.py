import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import Normalizer

#Iteration 1:
#I do not have Pipeline
#I do not have Cross-validation
#I do not have timing
#I do not output nicely
def run_cnn(trainX, trainY, testX, testY, #data
            confusion_matrix, classification_name, model_name, #paths
            batch_size=32,  epochs=10): #hyperparameters

    #PROBLEM -- Right now, we don't use Pipeline.
    #PROBLEM WITH PIPELINE >> NORMALIZATION CHECKS FOR MORE DIMENSIONS. WE DONT CARE ABOUT THAT. WE NEED SKLEARN 0.23.2
    # model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # model_pipeline = Pipeline(steps=[
    #     ("normalizer", Normalizer()),
    #     ("classifier", model)])
    # _ = model_pipeline.fit(trainX, trainY)

    model_pipeline = create_model()
    _ = model_pipeline.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)
    

    #Accuracy
    _, acc = model_pipeline.evaluate(testX, testY, verbose=0, batch_size=None)
    print('Accuracy: %.3f' % (acc * 100.0))
    #give classification report for cnn

    y_pred = model_pipeline.predict(testX, batch_size=None, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    #Classication report
    #cm = metrics.confusion_matrix(testY, y_pred.argmax(axis=1))
    print(metrics.classification_report(testY, y_pred_classes))
    
    #Write classification report to file
    with open(classification_name, 'w') as f:
        f.write(metrics.classification_report(testY, y_pred_classes))

    #plot confusion matrix
    cm = metrics.confusion_matrix(testY, y_pred_classes)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc)
    plt.title(all_sample_title, size = 15)
    plt.savefig(confusion_matrix)
    
    #save model
    model_pipeline.save(model_name)


def run_cnn_with_loaded_model(trainX, trainY, testX, testY, #data
            confusion_matrix, classification_name, model_name, #paths
            batch_size=32,  epochs=10): #hyperparameters

    #PROBLEM -- Right now, we don't use Pipeline.
    #PROBLEM WITH PIPELINE >> NORMALIZATION CHECKS FOR MORE DIMENSIONS. WE DONT CARE ABOUT THAT. WE NEED SKLEARN 0.23.2
    # model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # model_pipeline = Pipeline(steps=[
    #     ("normalizer", Normalizer()),
    #     ("classifier", model)])
    # _ = model_pipeline.fit(trainX, trainY)

    model_pipeline = keras.models.load_model(model_name)
    _ = model_pipeline.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)
    

    #Accuracy
    _, acc = model_pipeline.evaluate(testX, testY, verbose=0, batch_size=None)
    print('Accuracy: %.3f' % (acc * 100.0))
    #give classification report for cnn

    y_pred = model_pipeline.predict(testX, batch_size=None, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    #Classication report
    #cm = metrics.confusion_matrix(testY, y_pred.argmax(axis=1))
    print(metrics.classification_report(testY, y_pred_classes))
    
    #Write classification report to file
    with open(classification_name, 'w') as f:
        f.write(metrics.classification_report(testY, y_pred_classes))

    #plot confusion matrix
    cm = metrics.confusion_matrix(testY, y_pred_classes)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc)
    plt.title(all_sample_title, size = 15)
    plt.savefig(confusion_matrix)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



# Grave of the code that I tried to make work#
#                                            #
#                                            #
#                                            #
#Iteration two should have something from this grave   #

# def kfold_on_model(dataX, dataY, n_folds=5, epochs=10, batch_size=32):
#     scores = list()
#     # prepare cross validation
#     kfold = KFold(n_folds, shuffle=True, random_state=1)
#     # enumerate splits
#     for train_ix, test_ix in kfold.split(dataX):
#         # define model
#         model = create_model()
#         # select rows for train and test
#         trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
#         # fit model
#         _ = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
#                       validation_data=(testX, testY), verbose=0)
#         # evaluate model
#         _, acc = model.evaluate(testX, testY, verbose=0)
#         # stores scores
#         scores.append(acc)
#     return scores

# def summarize_performance(scores):
# 	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores), std(scores), len(scores)))


# def cross_validate_model(trainX, trainY, epochs=10, batch_size=32):
#     scores = kfold_on_model(trainX, trainY, epochs=epochs, batch_size=batch_size)
#     summarize_performance(scores)

# Do not call this function!! For some reasons it freezes the computer. I don't know why, but in theory it should work
# I have taken inspiration from this
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# This function uses keras and sklearn to train the model with grid search.

# def optimized_cnn_model(trainX, trainY, testX, testY):
# 	model_hyperparams = {
# 		'batch_size': [32, 64],
# 		'epochs': [10, 20]
# 	}

# 	model = KerasClassifier(model=create_model, verbose=0)

# 	grid_search_model = GridSearchCV(
# 	    estimator=model, param_grid=model_hyperparams, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# 	grid_search_model.fit(trainX, trainY)

# 	best_model = grid_search_model.best_estimator_.fit(trainX, trainY)

# 	print('Best Model Parameters: ', grid_search_model.best_params_)
# 	# acc
# 	_, acc = best_model.evaluate(testX, testY, verbose=0)
# 	print('Accuracy: %.3f' % (acc * 100.0))
