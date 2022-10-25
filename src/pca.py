from sklearn import decomposition
#function to perform pca, takes in the data and the number of components and returns the data after pca
def pca(data, n_components = 2):
    pca = decomposition.PCA(n_components)
    fitted_data = pca.fit(data)
    return fitted_data.transform(data)