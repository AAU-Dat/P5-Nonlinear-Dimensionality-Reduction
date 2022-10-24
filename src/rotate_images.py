import numpy as np
from scipy.ndimage import rotate
 
# make a function that takes image at a time and sends it to the function that rotates it, it gets a number of images, it should return all the rotated images. the degree of rotation should not be under 20 degrees and not over 360 degrees.
def rotate_image(matrix, degree):
    if degree <= 20:
        print("The degree is too small")
        return matrix
    else:
        rotated_matrix = rotate(matrix, degree, reshape=False).astype(int)
        return rotated_matrix
