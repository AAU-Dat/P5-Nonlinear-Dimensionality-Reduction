import load_mnist
import validation_data
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda():
    (train_data, val_data, test_data) = validation_data.divide_data(load_mnist.load_mnist())
    clf = LinearDiscriminantAnalysis()
    clf.fit_transform(train_data[0], train_data[1])
    print(clf.score(val_data[0], val_data[1]))

lda()




