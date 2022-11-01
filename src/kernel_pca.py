from sklearn import decomposition


# function to perform pca, takes in the data and the number of components and returns the data after pca
def kernel_pca(data, n_components=2):
    (train_data, test_data) = data
    kernel_pca = decomposition.KernelPCA(
        n_components=n_components, kernel="rbf")
    #Fits training data to the kernel pca model
    #fitted_train_data = kernel_pca.fit(train_data[0], train_data[1])
    #Transforms the training data to the new space
    kernel = kernel_pca.fit_transform(train_data[0])
    #kernel_pca_train_data = kernel_pca.transform(fitted_train_data, train_data[1])
    #fits the test data to the kernel pca model
    fitted_test_data = kernel_pca.fit(test_data[0], test_data[1])
    #Transforms the test data to the new space
    kernel_pca_test_data = (fitted_test_data.transform(test_data[0]), test_data[1])
    #returns the transformed data as a tuple
    return (kernel_pca_train_data, kernel_pca_test_data)
