import numpy as np
import gzip
import requests
import shutil

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
    

def download_files():
    curl_links = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz','http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz','http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    gz_file_names = ['train_file_image.gz', 'train_file_label.gz', 'test_file_image.gz', 'test_file_label.gz']
    file_names = ['train_file_image', 'train_file_label', 'test_file_image', 'test_file_label']

    for i in range (0,4):
        res = requests.get(curl_links[i])

        train_file_image = open(gz_file_names[i], 'wb')
        train_file_image.write(res.content)
        train_file_image.close()
    
        with gzip.open(gz_file_names[i], "rb") as f_in:
            with open(file_names[i], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

