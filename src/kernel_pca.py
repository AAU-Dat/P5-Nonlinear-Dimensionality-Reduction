from sklearn import decomposition
import numpy as np

# function to perform pca, takes in the data and the number of components and returns the data after pca
def kernel_pca(data, n_components=2):
    (train_data, test_data) = data
    kpca = decomposition.KernelPCA(kernel="rbf",n_components=2 , gamma=1)
    part_train_data = np.array_split(train_data[0], 6)
    part_train_labels = np.array_split(train_data[1], 6)
    i = 0
    kernel_pca_train_data = np.empty(60000, n_components)
    train_sd = np.array()
    while(i < 6):
        fitted_train_data = kpca.fit(part_train_data[i], part_train_labels[i])
        train_kpca = fitted_train_data.transform(part_train_data[i]) 
        
        train_sd = np.append(train_sd, train_kpca)

        #kernel_pca_train_data = np.append(kernel_pca_train_data, train_kpca)
        print("i = ", i)
        i += 1
    
    print("kernel_pca_train_data = ", kernel_pca_train_data.shape)
    change_tuple_firstvalue(data, kernel_pca_train_data)


    part_test_data = np.array_split(test_data[0], 2)
    part_test_labels = np.array_split(test_data[1], 2)

    fitted_test_data = kpca.fit(part_test_data[0], part_test_labels[0])

    test_kpca1 = (fitted_test_data.transform(part_test_data[0]), part_test_labels[0])
    
    fitted_test_data = kpca.fit(part_test_data[1], part_test_labels[1])
    test_kpca2 = (fitted_test_data.transform(part_test_data[1]), part_test_labels[1])

    kernel_pca_test_data = (np.concatenate((test_kpca1[0], test_kpca2[0])), np.concatenate((test_kpca1[1], test_kpca2[1])))
    print("kernel_pca_test_data = ", kernel_pca_test_data[0].shape)
    return (kernel_pca_train_data, kernel_pca_test_data)



 def change_tuple_firstvalue (data, new_data):
     new_training_data_tuble = (new_data, data[0][1])
     new_tuble = (new_training_data_tuble, data[1])
     return new_tuble