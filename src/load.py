
#Assignment: Load data from MNIST
#Most people use keras or tensorflow
#We do it manually, sort of


# We need a path for taking the data.
# We need the data. It is stored in MNIST folder. It should be made modular so as to work with MNIST+

# We need to load images. Images have a number and
# We need to load labels

import gzip
import numpy as np

def load_mnist_image(path):

    with gzip.open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(), np.uint8, offset=16)
    #now we have a long vector

    reshaped_image_data = image_data.reshape((-1, 784))
    #we convert the data into a float
    return reshaped_image_data / np.float32(256)
    
#Add index for size of file ([start_index:end_index])
def load_mnist_label(path,):
    with gzip.open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(), np.uint8, offset=8)
    #data does not need to be reshaped as it is already a vector
    return image_data
    
#The functions np.asanyarray do not seem to have an effect on the data
def load_mnist_all(path_of_img, path_of_lab, path_of_test_img, path_of_test_lab, validation_samples=10000):
    training_images = load_mnist_image(path_of_img)
    training_labels = load_mnist_label(path_of_lab)

    testing_images = load_mnist_image(path_of_test_img)
    testing_labels = load_mnist_label(path_of_test_lab)

    #validation gets x training samples
    
    training_images, validation_images = training_images[:-validation_samples], training_images[-validation_samples:]
    training_labels, validation_labels = training_labels[:-validation_samples], training_labels[-validation_samples:]
    
    # training_images = np.asanyarray(training_images, dtype=np.float32)
    # training_labels = np.asanyarray(training_labels, dtype=np.int32)
    
    training_data = (training_images, training_labels)

    # validation_images = np.asanyarray(validation_images, dtype=np.float32)
    # validation_labels= np.asanyarray(validation_labels, dtype=np.int32)
    validation_data = (validation_images, validation_labels)

    # testing_images = np.asanyarray(testing_images, dtype=np.float32)
    # testing_labels = np.asanyarray(testing_labels, dtype=np.int32)
    testing_data = (testing_images, testing_labels)
    return (training_data, validation_data, testing_data)

