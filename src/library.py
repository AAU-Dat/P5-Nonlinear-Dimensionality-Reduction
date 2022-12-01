import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# The results functions are similar for each dimensionality reduction method. The only difference is the pipeline and the hyperparameters:
# 1. Build the pipeline
# 2. Build the grid search
# 3. Perform grid search
# 4. Fit the final pipeline
# 5. Predict the results from the test set
# 6. Save the results
# Currently:
# The pipeline uses an ovo classifier with a SVC model. This seems to be better than LinearSVC, but it is hard to find information on LinearSVC in general.
# GridSearchCV scores are based on the f1_macro score
# GridSearchCV cross validation is stratisfied 5-fold.

# Add this for repeated stratified k-fold cross validation insteaf of standard stratified k-fold cross validation
# from sklearn.model_selection import RepeatedStratifiedKFold
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# search = GridSearchCV(pipeline, hyperparameters, cv=cv, scoring="f1_macro", verbose=10, n_jobs=-1)

# This function gets the results of the svm model without dimensionality reduction.


def baseline_svm_results(X, y, X_test, y_test, hyperparameters, methodname="baseline_svm"):
    baseline_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="linear", decision_function_shape="ovo", random_state=42))]
    )

    search = GridSearchCV(baseline_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=10,
                          n_jobs=-1)

    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    save_results(methodname,
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def pca_svm_results(X, y, X_test, y_test, hyperparameters, methodname="pca_svm"):
    # Build the pipeline. StandardScaler removes the mean and scales each feature/variable to unit variance (REQUIRED for PCA)
    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full", random_state=42)),
        ("classifier", SVC(kernel="linear", decision_function_shape="ovo", random_state=42))]
    )

    search = GridSearchCV(pca_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=10,
                          n_jobs=-1)

    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    save_results(methodname,
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def lda_svm_results(X, y, X_test, y_test, hyperparameters, methodname="lda_svm"):
    lda_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ("classifier", SVC(kernel="linear", decision_function_shape="ovo", random_state=42))]
    )

    search = GridSearchCV(lda_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=10,
                          n_jobs=-1)

    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    save_results(methodname,
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def isomap_svm_results(X, y, X_test, y_test, hyperparameters, methodname="isomap_svm"):
    # StandardScaler is not required for isomap, but HUGELY affects speed. Perhaps try running the pipeline with data of type float32 or float16.
    isomap_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("isomap", Isomap()),
        ("classifier", SVC(kernel="linear", decision_function_shape="ovo", random_state=42))]
    )

    search = GridSearchCV(isomap_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=10,
                          n_jobs=1)

    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    save_results(methodname,
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def kernel_pca_svm_results(X, y, X_test, y_test, hyperparameters, methodname="kernel_pca_svm"):
    kernel_pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kernel_pca", KernelPCA()),
        ("classifier", SVC(kernel="linear", decision_function_shape="ovo", random_state=42))]
    )

    search = GridSearchCV(kernel_pca_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=10,
                          n_jobs=1)

    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    save_results(methodname,
                 search.cv_results_,
                 ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)),
                 classification_report(y_test, y_pred, output_dict=True)
                 )


def save_results(methodname, results, confusion_matrix, classification_report):
    pd.DataFrame(results).to_csv(
        "src/results/cross_validation_" + methodname + ".csv",
        index=False)
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
