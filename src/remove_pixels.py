from textwrap import fill
import numpy as np
import random

#make function to remove random spaces from matrix and returns a new matrix with the spaces removed
def remove_pixels(matrix, procent = 10):
    #make a list of random numbers
    new_matrix = matrix.copy()
    random_list = []
    number_of_pixels = int((procent / 100) * 784)
    for i in range(0, number_of_pixels):
        random_list.append(random.randint(0, 783))
    
    #removes dublikates
    random_list = list(set(random_list))
    
    for i in range(0, len(random_list)):
        new_matrix[random_list[i]] = 0
    return new_matrix
