from sklearn.manifold import Isomap

# The most important parameters to consider are: data, n_neighbors and n_components

# returns a n_components array of shape (n_samples, n_components)
def reduced_data_with_isomap(data, n_components=2, n_neighbors=8):
    isomap = Isomap(n_components=n_components)
    return (isomap.fit_transform(data[0]), data[1])
