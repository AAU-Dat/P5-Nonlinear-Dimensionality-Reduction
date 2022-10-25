def create_dev_set(data):
    (train_data, test_data) = data
    
    number_of_train_samples = len(train_data[0])

    ratio = 1/6
    validation_size = int(number_of_train_samples * ratio)

    validation_data = (train_data[0][-validation_size:], train_data[1][-validation_size:])

    return (train_data, validation_data, test_data)
 