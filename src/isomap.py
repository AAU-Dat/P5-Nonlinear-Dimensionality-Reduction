from sklearn.manifold import Isomap
import numpy as np
# The most important parameters to consider are: data, n_neighbors and n_components

# returns a n_components array of shape (n_samples, n_components)
def reduced_data_with_isomap(data, n_components=2, n_neighbors=8):
    (train_data, test_data) = data
    isomap = Isomap(n_components = n_components, n_neighbors=n_neighbors)
    
    part_train_data = np.array_split(train_data[0], 4)
    part_train_labels = np.array_split(train_data[1], 4)

    part_test_data = np.array_split(test_data[0], 2)
    part_test_labels = np.array_split(test_data[1], 2)


    #Isomap on training data
    fitted_train_data = isomap.fit(part_train_data[0], part_train_labels[0])
    train_isomap = (fitted_train_data.transform(part_train_data[0]), part_train_labels[0]) 

    fitted_train_data = isomap.fit(part_train_data[1], part_train_labels[1])
    train_isomap1 = (fitted_train_data.transform(part_train_data[1]), part_train_labels[1]) 

    fitted_train_data = isomap.fit(part_train_data[2], part_train_labels[2])
    train_isomap2 = (fitted_train_data.transform(part_train_data[2]), part_train_labels[2]) 

    fitted_train_data = isomap.fit(part_train_data[3], part_train_labels[3])
    train_isomap3 = (fitted_train_data.transform(part_train_data[0]), part_train_labels[0]) 

    isomap_train_data = (np.concatenate((train_isomap[0], train_isomap1[0], train_isomap2[0], train_isomap3[0])), np.concatenate((train_isomap[1], train_isomap1[1], train_isomap2[1], train_isomap3[1])))

    #Isomap for test data
    fitted_test_data = isomap.fit(part_test_data[0], part_test_labels[0])
    test_isomap1 = (fitted_test_data.transform(part_test_data[0]), part_test_labels[0])
    
    fitted_test_data = isomap.fit(part_test_data[1], part_test_labels[1])
    test_isomap2 = (fitted_test_data.transform(part_test_data[1]), part_test_labels[1])

    isomap_test_data = (np.concatenate((test_isomap1[0], test_isomap2[0])), np.concatenate((test_isomap1[1], test_isomap2[1])))
    return (isomap_train_data, isomap_test_data)
