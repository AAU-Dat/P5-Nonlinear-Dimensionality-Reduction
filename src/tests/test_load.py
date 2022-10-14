
import sys
sys.path.append('../')
import load
import pre_process
import shutil
import os
# This function is written so as to check that the files are in the directory when written

def test_load_one_training_sample():
    load.download_files()
    number_of_samples = 1
    (training_data, test_data) = pre_process.load_mnist_all("train_file_image",
                                                            "train_file_label",
                                                            "test_file_image",
                                                            "test_file_label", number_of_samples)
    training_image = training_data[0]
    training_label = training_data[1]

    assert len(training_image) == 28 * 28  # number of pixels in an image
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

    assert len(training_image) == 28 * 28 * number_of_samples  # two images
    assert len(training_label) == number_of_samples