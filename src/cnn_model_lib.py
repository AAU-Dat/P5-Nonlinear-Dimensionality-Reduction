#Imports
import idx2numpy
import tensorflow as tf
from keras import layers, models
import matplotlib as plt
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import (PCA, KernelPCA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.preprocessing import (StandardScaler, Normalizer)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay)
from library import save_results
import seaborn as sns

def pca_cnn_results(X, y, X_test, y_test, hyperparameters):

    model = KerasClassifier(build_fn=cnn_model, epochs=10, verbose=0)

    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full")),
        ("unflatten", UnflattenTransformer()),
        ("classifier", model)
    ])

    search = GridSearchCV(pca_model_pipeline,
                            hyperparameters,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)
    
    search.fit(X, y)

    y_pred = search.best_estimator_.predict(X_test)
    save_results("pca_cnn",
        search.cv_results_, 
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
        classification_report(y_test, y_pred, output_dict=True))

def lda_cnn_results(X, y, X_test, y_test, hyperparameters):

    model = KerasClassifier(build_fn=cnn_model, epochs=10, verbose=0)

    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lda", LDA()),
        ("unflatten", UnflattenTransformer()),
        ("classifier", model)
    ])

    search = GridSearchCV(pca_model_pipeline,
                            hyperparameters,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)
    
    search.fit(X, y)

    y_pred = search.best_estimator_.predict(X_test)
    save_results("lda_cnn",
        search.cv_results_, 
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
        classification_report(y_test, y_pred, output_dict=True))

def kernel_pca_cnn_results(X, y, X_test, y_test, hyperparameters):

    model = KerasClassifier(build_fn=cnn_model, epochs=10, verbose=0)

    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kernelpca", KernelPCA()),
        ("unflatten", UnflattenTransformer()),
        ("classifier", model)
    ])

    search = GridSearchCV(pca_model_pipeline,
                            hyperparameters,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)
    
    search.fit(X, y)

    y_pred = search.best_estimator_.predict(X_test)
    save_results("kernel_pca_cnn",
        search.cv_results_, 
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
        classification_report(y_test, y_pred, output_dict=True))

def isomap_cnn_results(X, y, X_test, y_test, hyperparameters):

    model = KerasClassifier(build_fn=cnn_model, epochs=10, verbose=0)

    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("isomap", Isomap()),
        ("unflatten", UnflattenTransformer()),
        ("classifier", model)
    ])

    search = GridSearchCV(pca_model_pipeline,
                            hyperparameters,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)
    
    search.fit(X, y)

    y_pred = search.best_estimator_.predict(X_test)
    save_results("isomap_cnn",
        search.cv_results_, 
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
        classification_report(y_test, y_pred, output_dict=True))

def cnn_results(X, y, X_test, y_test, hyperparameters):

    model = KerasClassifier(build_fn=cnn_model, epochs=10, verbose=0)

    pca_model_pipeline = Pipeline(steps=[
        ("normalizer", Normalizer()),
        ("unflatten", UnflattenTransformer()),
        ("classifier", model)
    ])

    search = GridSearchCV(pca_model_pipeline,
                            hyperparameters,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)
    
    search.fit(X, y)

    y_pred = search.best_estimator_.predict(X_test)
    save_results("cnn_only",
        search.cv_results_, 
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
        classification_report(y_test, y_pred, output_dict=True))

#Transformer class for reshaping the data into a 4D array so that it can be used in the CNN model
class UnflattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        size = int(sqrt(X.shape[1]))
        return X.reshape(-1, size, size, 1)

#Convolutional Neural Network model
def cnn_model(init_mode='he_uniform', input_shape=(28,28,1), pool_type='max'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=init_mode, input_shape=input_shape, padding='same'))
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
    if pool_type == 'avg':
        model.add(layers.AveragePooling2D((2,2), padding='same'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
    if pool_type == 'avg':
        model.add(layers.AveragePooling2D((2,2), padding='same'))    
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, kernel_initializer=init_mode, activation='relu'))
    model.add(layers.Dense(10, kernel_initializer=init_mode))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
