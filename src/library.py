import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# Add this for repeated stratified k-fold cross validation insteaf of standard stratified k-fold cross validation
# from sklearn.model_selection import RepeatedStratifiedKFold
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# search = GridSearchCV(pipeline, hyperparameters, cv=cv, scoring="f1_macro", verbose=3, n_jobs=-1)


def pca_results(X, y, X_test, y_test, hyperparameters):
    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full")),
        ("classifier", OneVsOneClassifier(SVC(random_state=42)))]  # OneVsOne might be superfluous. LinearSVC might be better
    )

    search = GridSearchCV(pca_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=3,
                          n_jobs=-1)

    search.fit(X, y)
    final_pipeline = search.best_estimator_.fit(X, y)
    y_pred = final_pipeline.predict(X_test)

    save_results("pca",
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def lda_results(X, y, X_test, y_test, hyperparameters):
    lda_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ("classifier", OneVsOneClassifier(SVC(random_state=42)))]
    )

    search = GridSearchCV(lda_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=3,
                          n_jobs=-1)

    search.fit(X, y)
    final_pipeline = search.best_estimator_.fit(X, y)
    y_pred = final_pipeline.predict(X_test)

    save_results("lda",
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def isomap_results(X, y, X_test, y_test, hyperparameters):
    isomap_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("isomap", Isomap()),
        ("classifier", OneVsOneClassifier(SVC(random_state=42)))]
    )

    search = GridSearchCV(isomap_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=3,
                          n_jobs=-1)

    search.fit(X, y)
    final_pipeline = search.best_estimator_.fit(X, y)
    y_pred = final_pipeline.predict(X_test)

    save_results("isomap",
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def kernel_pca_results(X, y, X_test, y_test, hyperparameters):
    kernel_pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kernel_pca", KernelPCA()),
        ("classifier", OneVsOneClassifier(SVC(random_state=42)))]
    )

    search = GridSearchCV(kernel_pca_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=3,
                          n_jobs=-1)

    search.fit(X, y)
    final_pipeline = search.best_estimator_.fit(X, y)
    y_pred = final_pipeline.predict(X_test)

    save_results("kernel_pca",
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def save_results(methodname, results, confusion_matrix, classification_report):
    pd.DataFrame(results).to_csv(
        "src/results/cross_validation_" + methodname + ".csv")
    confusion_matrix.plot(cmap=plt.cm.Blues)
    plt.savefig("src/results/confusion_matrix_" +
                methodname + ".png", bbox_inches="tight")
    pd.DataFrame(classification_report).transpose().style.to_latex("src/results/classification_report_" + methodname + ".tex",
                                                                   caption="Classification report for " + methodname,
                                                                   label="tab:classification-report-" + methodname,
                                                                   position_float="centering",
                                                                   position="htb!",
                                                                   hrules=True,
                                                                   )
