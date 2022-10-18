import sys
sys.path.append('../')
import numpy as np
from sklearn.manifold import Isomap
import isomap


#Test for isomap where I reduce a 3x3 matrix to a 2x2 matrix
def test_isomap_function():
    n_neighbors = 2
    n_components = 2
    data = [[1, 2,3], [5, 6,7], [7, 8,9]]
    new_data = isomap.reduce_data_with_isomap(data=data, n_neighbors=n_neighbors, n_components=n_components)
    assert new_data.shape == (3,2)
    