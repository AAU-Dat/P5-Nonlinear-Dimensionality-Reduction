from sklearn import decomposition

def pca(data, n_components):
    pca = decomposition.PCA(n_components)
    pca.fit(data)
    return pca