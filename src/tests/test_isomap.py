import sys
sys.path.append('../')
import isomap
import numpy as np

def test_isomap():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    reduced_matrix = reduce_isomap(matrix=matrix, n_neighbors=5, n_components=2)

    assert reduced_matrix.shape == (2, 2)
