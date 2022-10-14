
import sys
sys.path.append('../')
import load
import pre_process
import shutil
import os

# This function is written so as to delete the files
def delete_files():
    os.remove('train_file_image')
    os.remove('train_file_label')
    os.remove('test_file_image')
    os.remove('test_file_label')
    os.remove('train_file_image.gz')
    os.remove('train_file_label.gz')
    os.remove('test_file_image.gz')
    os.remove('test_file_label.gz')
    if os.path.isdir('__pychache__'):
        shutil.rmtree('__pychache__')
    if os.path.isdir('.pytest_cache'):
        shutil.rmtree('.pytest_cache')
# This function is written so as to check that the files are in the directory when written


def check_files_exist():

    return os.path.isfile('train_file_image') and os.path.isfile('train_file_label') and os.path.isfile('test_file_image') and os.path.isfile('test_file_label') and os.path.isfile('train_file_image.gz') and os.path.isfile('train_file_label.gz') and os.path.isfile('test_file_image.gz') and os.path.isfile('test_file_label.gz')


def test_files_downloaded_correctly():
    load.download_files()
    assert check_files_exist() == True


def test_load_one_training_sample():
    load.download_files()
    number_of_samples = 1
    (training_data, test_data) = pre_process.load_mnist_all("train_file_image",
                                                            "train_file_label",
                                                            "test_file_image",
                                                            "test_file_label", number_of_samples)
    training_image = training_data[0]
    training_label = training_data[1]

    assert len(training_image) == 784  # number of pixels in an image
    assert len(training_label) == number_of_samples


def test_load_two_training_samples():
    load.download_files()
    number_of_samples = 2
    (training_data, test_data) = pre_process.load_mnist_all("train_file_image",
                                                            "train_file_label",
                                                            "test_file_image",
                                                            "test_file_label", number_of_samples)
    training_image = training_data[0]
    training_label = training_data[1]

    assert len(training_image) == 784 * 2  # two images
    assert len(training_label) == number_of_samples

def test_files_are_deleted():
    load.download_files()
    delete_files()
    assert check_files_exist() == False