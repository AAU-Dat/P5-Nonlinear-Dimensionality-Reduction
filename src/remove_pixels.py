from textwrap import fill
from xxlimited import new
import numpy as np
import random

def remove_pixels(data, number_of_pixels = 10):
    #call remove_pixels_from_image for each image in the data and return the new data tuble
    training_data = data[0][0]

    new_data = training_data
    #for image in training_data:
    #    new_data = np.append(new_data,remove_pixels_from_image(image, number_of_pixels))

    new_data = np.append(new_data,remove_pixels_from_image(training_data[2], number_of_pixels))
        
    new_data = np.array(new_data)
    new_data = new_data.reshape(-1, 784)
    print(len(new_data))
    print(training_data[2])
    print("\n")
    print(new_data[60000])
    print(np.array_equal(training_data[2], new_data[60000]))
    return change_tuple_firstvalue(data, new_data)


#make function to remove random spaces from matrix and returns a new matrix with the spaces removed
def remove_pixels_from_image(matrix, number_of_pixels = 10):
    new_matrix = matrix
    number = (0, 10)
    #delete random pixels from the matrix
    for pixel in range(784):
        new_matrix = np.delete(new_matrix, random.randint(0, 0))
    return new_matrix

def change_tuple_firstvalue (data, new_data):
    new_training_data_tuble = (new_data, data[0][1])
    new_tuble = (new_training_data_tuble, data[1])
    return new_tuble
