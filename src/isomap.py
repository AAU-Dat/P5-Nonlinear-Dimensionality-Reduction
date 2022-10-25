from sklearn.manifold import Isomap
import numpy
#import PCA
from sklearn.decomposition import PCA


# The most important parameters to consider are: data, n_neighbors and n_components

# returns a n_components array of shape (n_samples, n_components)
def reduced_data_with_isomap(data, n_components=2, n_neighbors=8):
    isomap = Isomap(n_components=n_components)
    return (isomap.fit_transform(data[0]), data[1])

#This function is made in case we want to get the object isomap, which is like the previous function but it returns the object isomap instead of returning the reduced data tuple
def isomap_object_fitted_to_images(data, n_components=2):
    isomap = Isomap(n_components=n_components)
    return isomap.fit(data[0])
