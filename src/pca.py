from sklearn import decomposition
#function to perform pca, takes in the data and the number of components and returns the data after pca
def pca(data, n_components = 2):
    (train_data, test_data) = data
    pca = decomposition.PCA(n_components = n_components)

    #Fits training data to the pca model
    fitted_train_data = pca.fit(train_data[0], train_data[1])
    #Transforms the training data to the new space
    pca_train_data = (fitted_train_data.transform(train_data[0]), train_data[1])

    #fits the test data to the pca model
    fitted_test_data = pca.fit(test_data[0], test_data[1])
    #Transforms the test data to the new space
    pca_test_data = (fitted_test_data.transform(test_data[0]), test_data[1])
    #returns the transformed data as a tuple
    
    return (pca_train_data, pca_test_data)