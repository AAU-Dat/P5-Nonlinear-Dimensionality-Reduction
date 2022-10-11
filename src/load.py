
#Assignment: Load data from MNIST
#Most people use keras or tensorflow

# I need a path for taking the data.
# I need the data. It is stored in MNIST folder.

# I need to load images
# I need to load labels


import gzip
from os import getcwd
import numpy as np
#version one 

def load_mnist_image(path):

    with gzip.open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(), np.uint8, offset=16)
    #now we have a long vector

    #we reshape so as to get a 
    reshaped_image_data = image_data.reshape((-1, 28, 28, 1))

    print(reshaped_image_data.shape)
    #we convert the data into a float
    return reshaped_image_data / np.float32(256)
    

def load_mnist_label(path):

    with gzip.open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(), np.uint8, offset=8)
    #now we have a long vector
    #data does not need to be reshaped
    print(image_data)
    #we convert the data into a float
    return image_data
    

load_mnist_image("mnist-original-dataset/t10k-images-idx3-ubyte.gz")