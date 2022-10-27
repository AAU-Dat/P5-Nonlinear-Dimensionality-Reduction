from textwrap import fill
import numpy as np
import random

#make function to remove random spaces from matrix and returns a new matrix with the spaces removed
def remove_pixels(matrix, number_of_pixels = 70):
    new_matrix = matrix.copy()
    count = 0
    while count < number_of_pixels:
        #get random number
        random_number = random.randint(0, 784)
        if new_matrix[random_number] == 255 and count < number_of_pixels:
            new_matrix[random_number] = 0
            count += 1
    return new_matrix
