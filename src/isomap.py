from sklearn.manifold import Isomap
"""
Reduce the dimensionality of the data with Isomap
    n_neighbors is the number of neighbors to consider for each point for the manifold approximation
    n_components is the number of dimensions to reduce the data to
"""


def reduce_isomap(data,
                n_neighbors=5, 
                n_components=2,
                eigen_solver='auto', 
                tol=0, # tolerance for stopping criterion
                max_iter=None, # maximum number of iterations for the optimization
                path_method='auto', # Djiakstra or Floyd-Warshall
                neighbors_algorithm='auto', # algorithm to compute nearest neighbors
                n_jobs=None, # number of parallel jobs to run, would be good idea to put it at a high number since isomap is slow
                ):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    isomap.fit(data)
    return isomap.transform(data)