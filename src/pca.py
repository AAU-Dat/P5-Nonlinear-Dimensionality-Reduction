from sklearn import decomposition
#function to perform pca, takes in the data and the number of components and returns the data after pca
def pca(data, n_components):
    pca = decomposition.PCA(n_components)
    new_data = pca.fit(data)
    return new_data