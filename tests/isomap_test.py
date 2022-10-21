from src.isomap import reduce_data_with_isomap
import numpy

#Convert a 3 x 3 matrix into a 3 x 2 matrix with isomap
def test_isomap():
    data = [[1,2,3], [4,5,6], [7,8,9]]
    # The transformation of the list into a numpy array is necessary for the isomap function
    # In practice the data will be a numpy array
    numpy_data = numpy.array(data)

    reduced_data = reduce_data_with_isomap(data=numpy_data, n_components=2, n_neighbors=2)
    assert reduced_data.shape == (3,2)




