import gzip
import os
import shutil

import numpy as np
import requests


# Creates a folder for the MNIST data if it does not exist
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


# Checks if the files for the MNIST dataset exists, if not it downloads them
def check_for_files(file_names, path):
    for i in range(0, len(file_names)):
        if not os.path.exists(file_names[i]):
            download_files(file_names, path)
            break


# Downlaods files from the MNIST dataset and creates files for the images and labels
def download_files(file_names, path):
    curl_links = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    gz_file_names = ['train_file_image.gz', 'train_file_label.gz',
                     'test_file_image.gz', 'test_file_label.gz']

    for i in range(0, 4):
        res = requests.get(curl_links[i])
        train_file_image = open(path + "/" + gz_file_names[i], 'wb')
        train_file_image.write(res.content)
        train_file_image.close()
        # Unzip the files
        with gzip.open(path + "/" + gz_file_names[i], "rb") as f_in:
            with open(path + "/" + file_names[i], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    # removes the .gz files
    for gz_file_name in gz_file_names:
        os.remove(path + "/" + gz_file_name)


# Loads the images from the MNIST dataset
def load_mnist_image(path, file_name, number_of_images):
    image_pixels = 28 * 28
    image_training_offset = 16
    path = path + "/" + file_name

    with open(path, 'rb') as file:
        image_data = np.frombuffer(file.read(
            (image_pixels * number_of_images) + image_training_offset), np.uint8, offset=image_training_offset)
    return image_data


# Loads the labels from the MNIST dataset
def load_mnist_label(path, file_name, number_of_labels):
    label_offset = 8
    path = path + "/" + file_name

    with open(path, 'rb') as file:
        image_data = np.frombuffer(
            file.read(number_of_labels + label_offset), np.uint8, offset=label_offset)
    return image_data


# Load all of the data from MNIST and return a tuple containing the training data, training labels, test data, and test labels
def load_mnist_all(path, file_names, number_of_samples):
    training_images = load_mnist_image(path, file_names[0], number_of_samples)
    training_labels = load_mnist_label(path, file_names[1], number_of_samples)

    # Consider changing this to scale eks. 1:6 for traing and test set
    if number_of_samples < 10000:
        testing_images = load_mnist_image(
            path, file_names[2], number_of_samples)
        testing_labels = load_mnist_label(
            path, file_names[3], number_of_samples)
    else:
        testing_images = load_mnist_image(path, file_names[2], 10000)
        testing_labels = load_mnist_label(path, file_names[3], 10000)

    training_data = (training_images, training_labels)
    testing_data = (testing_images, testing_labels)

    return (training_data, testing_data)


# reshape the data to be in the correct format and returns a tuble
def reshape_data(data):
    train_data, test_data = data
    reshaped_train_data = (train_data[0].reshape(-1, 784), train_data[1])
    reshaped_test_data = (test_data[0].reshape(-1, 784), test_data[1])
    return (reshaped_train_data, reshaped_test_data)


def normalize_data(data):
    train_data, test_data = data
    normalized_train_data = (np.round(train_data[0] / 255, 2), train_data[1])
    normalized_test_data = (np.round(test_data[0] / 255, 2), test_data[1])
    return (normalized_train_data, normalized_test_data)
