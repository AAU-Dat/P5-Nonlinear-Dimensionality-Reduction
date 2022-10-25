#Validation from training data
#Find the ratio beetween training and test
def create_dev_set(data):
    (train_data, test_data) = data
    
    number_of_train_samples = len(train_data[0]) 

    number_of_test_samples = len(test_data[0]) 
    
    ratio = number_of_test_samples /( number_of_train_samples + number_of_test_samples)

    validation_size = int(number_of_train_samples * ratio) 

    validation_data = (train_data[0][-validation_size:], train_data[1][-validation_size:])

    train_data = (train_data[0][0:number_of_train_samples-validation_size], train_data[1][0:number_of_train_samples-validation_size])

    return (train_data, validation_data, test_data)
 