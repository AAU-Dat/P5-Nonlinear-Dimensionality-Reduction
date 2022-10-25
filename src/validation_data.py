#Divide the training data set into a training and validation set

def divide_data(data, validation_size=10000, ratio=0.8):
    (train_data, test_data) = data
    number_of_train_samples = len(train_data[0])

    ratio_percentage = int(number_of_train_samples * ratio)


    #Validation set and test set have the same size if the we have less than 10000 samples
    if number_of_train_samples <= validation_size:
        validation_size = number_of_train_samples - ratio_percentage
        test_data =(test_data[0][ratio_percentage:], test_data[1][ratio_percentage:])        
        
    #Validation set and test set have the same size so as to have the 80 % 20 % ratio between train and validation set
    if (number_of_train_samples - ratio_percentage) < validation_size:
        validation_size = number_of_train_samples - ratio_percentage
        test_data =(test_data[0][-validation_size:], test_data[1][-validation_size:])
    
    validation_data = (train_data[0][-validation_size:], train_data[1][-validation_size:])

    train_data = (train_data[0][0:number_of_train_samples-validation_size], train_data[1][0:number_of_train_samples-validation_size])

    return (train_data, validation_data, test_data)
