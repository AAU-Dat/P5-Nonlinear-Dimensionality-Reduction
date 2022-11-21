import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_gallery(title, images, n_col=5, n_row=5, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape((28, 28)),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal",
                 shrink=0.99, aspect=40, pad=0.01)
    plt.show()


def plot_components(X, y):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i]),
                 fontdict={'size': 15})

    plt.xticks([]), plt.yticks([]), plt.ylim(
        [-0.1, 1.1]), plt.xlim([-0.1, 1.1])


size = 150
datafile = 'mnist_data/train_file_image'
targetfile = 'mnist_data/train_file_label'
testfile = 'mnist_data/test_file_image'
testtargetfile = 'mnist_data/test_file_label'
data = idx2numpy.convert_from_file(datafile).reshape(60000, 784)
targets = idx2numpy.convert_from_file(targetfile)
test = idx2numpy.convert_from_file(testfile).reshape(10000, 784)
testtargets = idx2numpy.convert_from_file(testtargetfile)
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

data = StandardScaler().fit_transform(data[:size])
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[:size])

# plt.figure()
# plt.scatter(data_pca[:size, 0], data_pca[:size, 1], c=targets, cmap="tab10")
# plt.colorbar()
# plt.show()

acc_list = []
pc_list = []


for component in range(2, 101):
    data = StandardScaler().fit_transform(data[:size])
    test_data = StandardScaler().fit_transform(test)
    pca_test = PCA(n_components=component)
    pca = PCA(n_components=component)
    data_pca = pca.fit_transform(data[:size])
    test_data_pca = pca_test.fit_transform(test_data)

    clf = svm.SVC(kernel='linear')
    clf.fit(data_pca[:size], targets[:size])
    predictions = clf.predict(test_data_pca)
    acc_list.append(accuracy_score(testtargets, predictions))
    pc_list.append(component)

plt.figure(figsize=[12, 9])
plt.scatter(pc_list, acc_list)
plt.title('SVM Accuarcy as a Function of Number of Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Accuracy')
