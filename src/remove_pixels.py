from textwrap import fill
import numpy as np
import random

def remove_pixels(data, number_of_pixels = 70):
    #call remove_pixels_from_image for each image in the data and return the new data tuble
    new_data = []
    for image in data:
        new_data.append(remove_pixels_from_image(image[0], number_of_pixels))
    return new_data

#make function to remove random spaces from matrix and returns a new matrix with the spaces removed
def remove_pixels_from_image(matrix, number_of_pixels = 70):
    new_matrix = matrix.copy()
    count = 0
    while count < number_of_pixels:
        #get random number
        random_number = random.randint(0, 783)
        if new_matrix[random_number] <= 0 and count < number_of_pixels:
            new_matrix[random_number] = 0
            count += 1
    return new_matrix
