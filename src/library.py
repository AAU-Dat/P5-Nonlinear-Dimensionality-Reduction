import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def pca_results(X, y, X_test, y_test, hyperparameters):
    pca_model_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full")),
        ("classifier", OneVsOneClassifier(SVC(kernel="linear", random_state=42)))]  # OneVsOne might be superfluous. LinearSVC might be better
    )
    search = GridSearchCV(pca_model_pipeline,
                          hyperparameters,
                          cv=5,
                          scoring="f1_macro",
                          verbose=2,
                          n_jobs=-1)

    search.fit(X, y)
    final_pipeline = search.best_estimator_.fit(X, y)
    y_pred = final_pipeline.predict(X_test)

    save_results("pca",
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
    pd.DataFrame(classification_report).transpose().to_latex("src/results/classification_report_" + methodname + ".tex",
                                                             caption="Classification report for " + methodname,
                                                             label="tab:" + methodname)
