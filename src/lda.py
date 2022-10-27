from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda(data):
    #data is a tuple of (training_data, test_data)
    (train_data, test_data) = data
    #creates the LDA model
    lda = LinearDiscriminantAnalysis()
    
    #Fits training data to the lda model
    fitted_train_data = lda.fit(train_data[0], train_data[1])
    #Transforms the training data to the new space
    lda_train_data = (fitted_train_data.transform(train_data[0]), train_data[1])
    
    #fits the test data to the lda model
    fitted_test_data = lda.fit(test_data[0], test_data[1])
    #Transforms the test data to the new space
    lda_test_data = (fitted_test_data.transform(test_data[0]), test_data[1])
    #returns the transformed data as a tuple
    
    return (lda_train_data, lda_test_data)



