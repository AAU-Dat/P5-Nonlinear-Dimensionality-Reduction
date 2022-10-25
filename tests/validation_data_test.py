import src.validation_data
import src.load_mnist

def test_divide_data_for_all_data():
    (train_data, test_data) = src.load_mnist.load_mnist()
    (train_data, validation_data, test_data) = src.validation_data.divide_data((train_data, test_data))
    assert len(train_data[0]) == 50000
    assert len(validation_data[0]) == 10000
    assert len(test_data[0]) == 10000

def test_divide_data_for_10000_data():
    (train_data, test_data) = src.load_mnist.load_mnist(10000)
    (train_data, validation_data, test_data) = src.validation_data.divide_data((train_data, test_data))
    assert len(train_data[0]) == 8000
    assert len(validation_data[0]) == 2000
    assert len(test_data[0]) == 2000

def test_divide_data_for_10000_data_with_different_ratio():
    (train_data, test_data) = src.load_mnist.load_mnist(10000)
    (train_data, validation_data, test_data) = src.validation_data.divide_data((train_data, test_data), ratio=0.5)
    assert len(train_data[0]) == 5000
    assert len(validation_data[0]) == 5000
    assert len(test_data[0]) == 5000

def test_divide_data_for_20000_data_with_different_ratio():
    (train_data, test_data) = src.load_mnist.load_mnist(20000)
    (train_data, validation_data, test_data) = src.validation_data.divide_data((train_data, test_data), ratio=0.5)
    assert len(train_data[0]) == 10000
    assert len(validation_data[0]) == 10000
    assert len(test_data[0]) == 10000

def test_divide_data_with_20000_samples():
    (train_data, test_data) = src.load_mnist.load_mnist(20000)
    (train_data, validation_data, test_data) = src.validation_data.divide_data((train_data, test_data))
    assert len(train_data[0]) == 16000
    assert len(validation_data[0]) == 4000
    assert len(test_data[0]) == 4000