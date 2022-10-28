
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import time
import numpy as np
    
def return_metrics_for_data(data, labels, start=time.time()):
    #we are not interested in the time it takes to compute without dim red
    if data.shape[1] < 784:    
        print("Elapsed time for dim. red: %.2f s" % (end_time(start)))

    data_shape = data.shape
    average_distance_for_data = average_distance(data)
    average_distance_between_classes_for_data = average_distance_between_classes(data, labels)
    return (data_shape, average_distance_between_classes_for_data, average_distance_for_data, average_distance_between_classes_for_data / average_distance_for_data)

#calculate the mean for each class
def mean_for_each_class_function(data, labels):
    mean_for_each_class = []
    for i in range(0, 10):
        mean_for_each_class.append(np.mean(data[labels == i], axis=0))
    return mean_for_each_class

#calculate average distance for all data points
def average_distance(data):
    average_distance = 0
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            average_distance += np.linalg.norm(data[i] - data[j])
    return average_distance / (len(data) * len(data))

#calculate the average distance between classes
def average_distance_between_classes(data, labels):
    mean_for_each_class = mean_for_each_class_function(data, labels)
    average_distance_between_classes = 0
    for i in range(0, 10):
        for j in range(0, 10):
            average_distance_between_classes += np.linalg.norm(mean_for_each_class[i] - mean_for_each_class[j])
    return average_distance_between_classes / 100

def end_time(start):
    return time.time() - start
