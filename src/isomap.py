from sklearn.manifold import Isomap

# The most important parameters to consider are: data, n_neighbors and n_components

# returns a n_components array of shape (n_samples, n_components)


def reduce_data_with_isomap(data,
                            n_neighbors=5,
                            n_components=2,
                            # number of parallel jobs to run; as isomap can be slow it is recommended to use some (or all)available cores
                            n_jobs=None,
                            ):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

    return isomap.fit_transform(data)
