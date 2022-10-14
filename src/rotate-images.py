# dont know if it works
# import the Python Image processing Library
from re import S
from PIL import Image   
import numpy as np
import cmath
def main():
    # open image butterfly.jpg
    # make 28x28 matrix
    matrix = np.empty((28,28))
    matrix 
    for i in range(0, 28):
        for j in range(0, 28):
            matrix[i][j] = j
    matrix = np.array(matrix)
    # matrix = np.array([[1, 2, 3], [4, 5, 6]])
    #print(matrix)
    #print('hello')
    rotate_image(matrix, 90)
    #print('hello1')
    #print(np.rot90(matrix))
    
# make a function that takes image at a time and sends it to the function that rotates it, it gets a number of images, it should return all the rotated images
def rotate_image(matrix, degree):

    rotated_matrix = np.empty((28,28)).fill(0)
    print(rotated_matrix)
    # make a function that rotates the matrix by a degree
    rotated_matrix = np.rot90(matrix, degree)
    for i in range(0, 28):
        for j in range(0, 28):
            location = rotate_pixel(np.array([i,j]), degree)
            if location[0] < 0 or location[0] > 27 or location[1] < 0 or location[1] > 27:
                a=1
            else:
                #print(location)
                rotated_matrix[np.round(location[0])][np.round(location[1])] = matrix[i][j] 
            #print(np.round(location[0].astype(int)), np.round(location[1].astype(int)))
    # return the rotated matrix
    return rotated_matrix

#takes a pixel and rotates it by a degree
def rotate_pixel(location, degree):
    matrix4 = np.array([[np.cos(degree), cmath.sinh(degree)], [cmath.sin(degree), cmath.cos(degree)]]) 
    rotated_pixel = np.matmul(location, matrix4)
    return rotated_pixel.astype(int)
# make a function that rotates the image by 45 degrees and saves it in a new image, it gets one image to rotate and returns the new images
if __name__ == '__main__':
    main()
# make test for the function that rotates an matrix by a degree
# def test_rotate_image():
#     # make a matrix
#     matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     # make a function that rotates the matrix by 45 degrees
#     rotated_matrix = rotate_image(matrix, 90)
#     # make a matrix that is the expected result
#     expected_matrix = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])
#     # compare the expected matrix with the rotated matrix
#     assert rotated_matrix == expected_matrix