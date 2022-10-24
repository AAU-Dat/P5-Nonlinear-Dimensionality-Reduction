from src.rotate_images import rotate_image  
import numpy as np

# tests if the function output is correct
def test_rotate_image():
    # make a matrix
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # make a function that rotates the matrix by 45 degrees
    rotated_matrix = rotate_image(matrix, 90)
    # make a matrix that is the expected result
    expected_matrix = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
    # compare the expected matrix with the rotated matrix
    for i in range(0, 3):
        for j in range(0, 3):
            assert rotated_matrix[i][j] == expected_matrix[i][j]