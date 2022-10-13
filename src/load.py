import numpy as np

image_pixels = 28 * 28
image_training_offset = 16
label_offset = 8
#Loads the images from the MNIST dataset
def load_mnist_image(path, number_of_images=60000):
    with open(path, 'rb') as file:
        image_data = np.frombuffer(file.read((image_pixels * number_of_images)+ image_training_offset), np.uint8, offset=image_training_offset)
    return image_data
    
#Loads the labels from the MNIST dataset
def load_mnist_label(path, number_of_labels=60000):
    with open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(number_of_labels + label_offset), np.uint8, offset=label_offset)
    return image_data
    
