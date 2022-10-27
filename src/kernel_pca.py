from sklearn import decomposition


# function to perform pca, takes in the data and the number of components and returns the data after pca
def pca(data, n_components=2):
    kernel_pca = decomposition.KernelPCA(
        n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
    )
    fitted_model = kernel_pca.fit(X_train).transform(X_test)
    return (fitted_model.transform(data), data[1])
