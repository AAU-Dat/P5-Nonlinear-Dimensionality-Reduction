import load 
# Load all of the data from MNIST and return a tuple containing the training data, training labels, test data, and test labels


def load_mnist_all(path_of_img, path_of_lab, path_of_test_img, path_of_test_lab, number_of_samples=60000):
    training_images = load.load_mnist_image(path_of_img, number_of_samples)
    training_labels = load.load_mnist_label(path_of_lab, number_of_samples)
    testing_images = load.load_mnist_image(path_of_test_img, number_of_samples)
    testing_labels = load.load_mnist_label(path_of_test_lab, number_of_samples)

    training_data = (training_images, training_labels)
    testing_data = (testing_images, testing_labels)

    return (training_data, testing_data)
