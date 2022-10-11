
#Assignment: Load data from MNIST
#Most people use keras or tensorflow

# I need a path for taking the data.
# I need the data. It is stored in MNIST folder.

# I need to load images
# I need to load labels


import gzip
import numpy as np
#version one 

def load_mnist_image(path):

    with gzip.open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(), np.uint8, offset=16)
    #now we have a long vector

    #we reshape so as to get a 
    reshaped_image_data = image_data.reshape((-1, 784))

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
    


def load_mnist_all():
    training_images = load_mnist_image("mnist-original-dataset/train-images-idx3-ubyte.gz")
    training_labels = load_mnist_label("mnist-original-dataset/train-labels-idx1-ubyte.gz")

    testing_images = load_mnist_image("mnist-original-dataset/t10k-images-idx3-ubyte.gz")
    testing_labels = load_mnist_image("mnist-original-dataset/t10k-labels-idx1-ubyte.gz")

    #validation gets 20000 training samples
    training_images, validation_images = training_images[:-20000], training_images[-20000:]
    training_labels, validation_labels = training_labels[:-20000], training_labels[-20000:]

    #prettify
    training_images = np.asanyarray(training_images, dtype=np.float32)
    training_labels = np.asanyarray(training_labels, dtype=np.int32)

    validation_images = np.asanyarray(validation_images, dtype=np.float32)
    validation_labels= np.asanyarray(validation_labels, dtype=np.int32)

    testing_images = np.asanyarray(testing_images, dtype=np.float32)
    testing_labels = np.asanyarray(testing_labels, dtype=np.int32)

    #It is a method by which I can give back
    return (training_images, training_labels), (validation_images, validation_labels), (testing_images, testing_labels)

load_mnist_all()