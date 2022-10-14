from sklearn.manifold import Isomap

"""
Reduce the dimensionality of the data with Isomap
    n_neighbors is the number of neighbors to consider for each point for the manifold approximation
    n_components is the number of dimensions to reduce the data to
"""


def reduce_isomap(data, n_neighbors=5, n_components=2):
    isomap = Isomap(n_neighbors, n_components)
    isomap.fit(data)
    return isomap.transform(data)
