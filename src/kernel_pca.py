from sklearn import decomposition
import numpy as np

# function to perform pca, takes in the data and the number of components and returns the data after pca
def kernel_pca(data, n_components=2):
    (train_data, test_data) = data
    kpca = decomposition.KernelPCA(kernel="rbf",n_components=n_components , gamma=1)

    part_train_data = np.array_split(train_data[0], 4)
    part_train_labels = np.array_split(train_data[1], 4)

    part_test_data = np.array_split(test_data[0], 2)
    part_test_labels = np.array_split(test_data[1], 2)
 
    #kernel pca on training data
    fitted_train_data = kpca.fit(part_train_data[0], part_train_labels[0])
    train_kpca = (fitted_train_data.transform(part_train_data[0]), part_train_labels[0]) 

    fitted_train_data = kpca.fit(part_train_data[1], part_train_labels[1])
    train_kpca1 = (fitted_train_data.transform(part_train_data[1]), part_train_labels[1]) 

    fitted_train_data = kpca.fit(part_train_data[2], part_train_labels[2])
    train_kpca2 = (fitted_train_data.transform(part_train_data[2]), part_train_labels[2]) 

    fitted_train_data = kpca.fit(part_train_data[3], part_train_labels[3])
    train_kpca3 = (fitted_train_data.transform(part_train_data[0]), part_train_labels[0]) 

    kernel_pca_train_data = (np.concatenate((train_kpca[0], train_kpca1[0], train_kpca2[0], train_kpca3[0])), np.concatenate((train_kpca[1], train_kpca1[1], train_kpca2[1], train_kpca3[1])))

    #Kernel pca for test data
    fitted_test_data = kpca.fit(part_test_data[0], part_test_labels[0])
    test_kpca1 = (fitted_test_data.transform(part_test_data[0]), part_test_labels[0])
    
    fitted_test_data = kpca.fit(part_test_data[1], part_test_labels[1])
    test_kpca2 = (fitted_test_data.transform(part_test_data[1]), part_test_labels[1])

    kernel_pca_test_data = (np.concatenate((test_kpca1[0], test_kpca2[0])), np.concatenate((test_kpca1[1], test_kpca2[1])))
    return (kernel_pca_train_data, kernel_pca_test_data)
