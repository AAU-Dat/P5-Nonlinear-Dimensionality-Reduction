import src.load_mnist

def test_load_mnist_size_of_dataset():
    (train, test) = src.load_mnist.load_mnist()
    assert train[0].shape == (60000, 784)
    assert test[0].shape == (10000, 784)

def test_load_mnist_size_of_single_image():
    (single_train, single_test) = src.load_mnist.load_mnist(1)
    assert single_train[0].shape == (1, 784)
    assert single_test[0].shape == (1, 784)

def test_load_mnist_custom_size_15000():
    (custom_train, custom_test) = src.load_mnist.load_mnist(15000)
    assert custom_train[0].shape == (15000, 784)
    assert custom_test[0].shape == (10000, 784)

#This is sus
def test_load_mnist_custom_size_with_zero():
    (custom_train, custom_test) = src.load_mnist.load_mnist(0)
    assert custom_train[0].shape == (0, 784)
    assert custom_test[0].shape == (0, 784)