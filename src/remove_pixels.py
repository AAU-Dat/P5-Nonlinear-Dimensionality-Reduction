from textwrap import fill
import numpy as np
import random

def remove_pixels(data, number_of_pixels = 70):
    new_data = []
    print(data[0])
    for i in range(len(data[0])):
        new_data.append(remove_pixels_from_image(data[0][i], number_of_pixels))
    return (new_data, data[1])

#make function to remove random spaces from matrix and returns a new matrix with the spaces removed
def remove_pixels_from_image(matrix, number_of_pixels):
    new_matrix = matrix.copy()
    count = 0
    while count < number_of_pixels:
        #get random number
        random_number_row = random.randint(0, 28)
        random_number_column = random.randint(0, 28)
        if new_matrix[random_number_column][random_number_row] <= 0 and count < number_of_pixels:
            new_matrix[random_number_column][random_number_row] = 0
            count += 1
    return new_matrix
