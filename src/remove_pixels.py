from textwrap import fill
from xxlimited import new
import numpy as np
import random

def remove_datapoints(data, num_of_datapoints):
    (training_data, test_data) = data
    list_training_data_img = list(training_data[0][0])
    i = 0
    while i < num_of_datapoints:
        random_index = random.randint(0, len(list_training_data_img) - 1)
        if (list_training_data_img[random_index] > 0):
            list_training_data_img[random_index] = 0
            i += 1
    training_data = (np.array(list_training_data_img), training_data[1][0])
    return (training_data, test_data)

# def remove_datapoints(data, num_of_datapoints):
#     (training_data, test_data) = data
#     list_training_data_img = list(training_data[0][0])
#     matrix = np.reshape(list_training_data_img, (28, 28))
#     num = [random.randint(0, 27), random.randint(0, 27)]
#     surrounding_matrix = surrounding(matrix, num, 1, None)
#     i = 0
#     # while i < num_of_datapoints:
#     #     random_index = [random.randint(0, 27), random.randint(0, 27)]
#     #     if (matrix[random_index[0], random_index[1]] > 0):
#     #         matrix[random_index[0], random_index[1]] = 0
            
#     #         i += 1
#     training_data = tuple(list_training_data_img)
#     return (training_data, test_data)

# def surrounding(x, idx, radius=1, fill=0):
#     """ 
#     Gets surrounding elements from a numpy array 
  
#     Parameters: 
#     x (ndarray of rank N): Input array
#     idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
#         the entire axis is returned.
#     radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis, 
#         the entire axis is returned.
#     fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
#         If value is None, only the surrounding indices that are within the original array are returned.
  
#     Returns: 
#     ndarray: The surrounding elements at the specified index
#     """
    
#     assert len(idx) == len(x.shape)
    
#     if np.isscalar(radius):
#         radius = tuple([radius for i in range(len(x.shape))])
#     slices = []
#     paddings = []
#     for axis in range(len(x.shape)):
#         if idx[axis] is None or radius[axis] is None:
#             slices.append(slice(0, x.shape[axis]))
#             paddings.append((0, 0))
#             continue
            
#         r = radius[axis]
#         l = idx[axis] - r 
#         r = idx[axis] + r
        
#         pl = 0 if l > 0 else abs(l)
#         pr = 0 if r < x.shape[axis] else r - x.shape[axis] + 1
        
#         slices.append(slice(max(0, l), min(x.shape[axis], r+1)))
#         paddings.append((pl, pr))

#     return x[slices]


# def remove_pixels(data, number_of_pixels = 10):
#     #call remove_pixels_from_image for each image in the data and return the new data tuble
#     training_data = data[0][0]

#     new_data = training_data
#     #for image in training_data:
#     #    new_data = np.append(new_data,remove_pixels_from_image(image, number_of_pixels))

#     new_data = np.append(new_data,remove_pixels_from_image(training_data[2], number_of_pixels))
        
#     new_data = np.array(new_data)
#     new_data = new_data.reshape(-1, 784)
#     print(len(new_data))
#     print(training_data[2])
#     print("\n")
#     print(new_data[60000])
#     print(np.array_equal(training_data[2], new_data[60000]))
#     return change_tuple_firstvalue(data, new_data)


# #make function to remove random spaces from matrix and returns a new matrix with the spaces removed
# def remove_pixels_from_image(matrix, number_of_pixels = 10):
#     new_matrix = matrix
#     number = (0, 10)
#     #delete random pixels from the matrix
#     for pixel in range(784):
#         new_matrix = np.delete(new_matrix, random.randint(0, 0))
#     return new_matrix

# def change_tuple_firstvalue (data, new_data):
#     list_training_data_img_tuble = (new_data, data[0][1])
#     new_tuble = (list_training_data_img_tuble, data[1])
#     return new_tuble


