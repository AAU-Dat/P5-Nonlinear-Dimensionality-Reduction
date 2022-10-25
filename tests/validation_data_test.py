import src.validation_data
import src.load_mnist

def test_create_new_data():
    (train_data, test_data) = src.load_mnist.load_mnist(60000)
    (train_data, validation_data, test_data) = src.validation_data.create_dev_set((train_data, test_data))
    assert len(train_data[0]) == 60000
    assert len(validation_data[0]) == 10000
    assert len(test_data[0]) == 10000

def test_create_new_data_with_10000_samples():
    (train_data, test_data) = src.load_mnist.load_mnist(10000)
    (train_data, validation_data, test_data) = src.validation_data.create_dev_set((train_data, test_data))
    assert len(train_data[0]) == 10000
    assert len(validation_data[0]) == 1666
    assert len(test_data[0]) == 10000
