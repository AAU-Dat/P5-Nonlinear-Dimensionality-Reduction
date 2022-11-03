
from textwrap import fill
from xxlimited import new
import numpy as np
from numpy.typing import ArrayLike
import random
import load_mnist as lm
import matplotlib.pyplot as plt

# def remove_datapoints(data, num_of_datapoints):
#     (training_data, test_data) = data
#     picture_one = training_data[0][0]
#     for i in range(num_of_datapoints):
#         picture_one = remove_random_pixel(picture_one)
#     return (picture_one, training_data[1][0])

# def remove_random_pixel(data):
#     pixel = data[random.randint(0, 783)]
#     if(pixel > 0):
#         pixel = 0
#     else:
#         remove_random_pixel(data)
#     return data

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
#     new_training_data_tuble = (new_data, data[0][1])
#     new_tuble = (new_training_data_tuble, data[1])
#     return new_tuble

def find_pixel_cluster (picture: ArrayLike):
    black_indixes = np.nonzero(picture)
    black_indixes = black_indixes[0]
    np.random.shuffle(black_indixes)
    for index in black_indixes:
        if(index +28 > 784):
            break
        if (picture[index + 1] != 0 and picture[index + 29] != 0 and picture[index + 28] != 0 ):
            picture = remove_pixel_cluster(picture, index)
            break

    return picture




def remove_pixel_cluster (picture: ArrayLike, index: int):
    picture[index] = 0
    picture[index + 1] = 0
    picture[index + 29] = 0 
    picture[index + 28] = 0 
    return picture

def test ():
    data = lm.load_mnist()
    training_pictures = data[0][0]
    new_training_pictures  = np.array(training_pictures, copy=True)
    for picture in new_training_pictures:
        new_training_pictures[0] = find_pixel_cluster(picture)

    plot_digits(new_training_pictures)

def test_wrapper():
    data = lm.load_mnist()
    remove_pixel_augmentation(data)
    print(data[0][0].shape)

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                            subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28),
                cmap='binary', interpolation='nearest',
                clim=(0, 16))
    fig.savefig('hello1')

def remove_pixel_augmentation (data: ArrayLike) -> ArrayLike:
    training_pictures = data[0][0]
    new_training_pictures  = np.array(training_pictures, copy=True)
    for picture in new_training_pictures:
        new_training_pictures[0] = find_pixel_cluster(picture)
    
    new_data = ((new_training_pictures, data[0][1]), data[1])
