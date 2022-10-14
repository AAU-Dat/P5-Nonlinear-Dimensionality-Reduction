import load 
import pre_process 
import numpy as np


def main():

    load.download_files()
    number_of_samples = 1
    (training_data, test_data) = pre_process.load_mnist_all("train_file_image",
                                                          "train_file_label",
                                                          "test_file_image",
                                                          "test_file_label", number_of_samples)
    
    
if __name__ == "__main__":
    main()
