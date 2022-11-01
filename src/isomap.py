from sklearn.manifold import Isomap

# The most important parameters to consider are: data, n_neighbors and n_components

# returns a n_components array of shape (n_samples, n_components)
def reduced_data_with_isomap(data, n_components=2, n_neighbors=8):
    (train_data, test_data) = data
    isomap = Isomap(n_components = n_components, n_neighbors=n_neighbors)
    
    #Fits training data to the isomap model
    fitted_train_data = isomap.fit(train_data[0], train_data[1])
    #Transforms the training data to the new space
    isomap_train_data = (fitted_train_data.transform(train_data[0]), train_data[1])

    #fits the test data to the isomap model
    fitted_test_data = isomap.fit(test_data[0], test_data[1])
    #Transforms the test data to the new space
    isomap_test_data = (fitted_test_data.transform(test_data[0]), test_data[1])
    #returns the transformed data as a tuple
    
    return (isomap_train_data, isomap_test_data)