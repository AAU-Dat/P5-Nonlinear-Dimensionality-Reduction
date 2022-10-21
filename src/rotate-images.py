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

# tests if the function output is correct
# def test_rotate_image():
#       # make a matrix
#       matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#       # make a function that rotates the matrix by 45 degrees
#       rotated_matrix = rotate_image(matrix, 90)
#       # make a matrix that is the expected result
#       expected_matrix = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])
#       # compare the expected matrix with the rotated matrix
#       for i in range(0, 3):
#          for j in range(0, 3):
#              assert rotated_matrix[i][j] == expected_matrix[i][j]