import load 
import pre_process 


def main():
    number_of_samples = 3
    (training_data, test_data) = pre_process.load_mnist_all("../mnist-original-dataset/train-images-idx3-ubyte",
                                                          "../mnist-original-dataset/train-labels-idx1-ubyte",
                                                          "../mnist-original-dataset/t10k-images-idx3-ubyte",
                                                          "../mnist-original-dataset/t10k-labels-idx1-ubyte", number_of_samples)

if __name__ == "__main__":
    main()
