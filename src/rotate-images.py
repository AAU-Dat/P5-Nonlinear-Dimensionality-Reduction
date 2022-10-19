import numpy as np
from scipy.ndimage import rotate

#def main():
    # open image butterfly.jpg
    # make 28x28 matrix
    #matrix = np.empty((10,10))
    #for i in range(0, 10):
    #    for j in range(0, 10):
    #        matrix[i][j] = j
    #print(matrix)
    #print(rotate_image(matrix, 90))
    #print('hello1')
    #test_rotate_image()
    #print(np.rot90(matrix))
    
# make a function that takes image at a time and sends it to the function that rotates it, it gets a number of images, it should return all the rotated images
def rotate_image(matrix, degree):
    #rotation = np.array([[np.cos(degree), -cmath.sin(degree)], [cmath.sin(degree), cmath.cos(degree)]]) 
    # rotated_matrix = np.empty((28,28)).fill(0)
    # print(rotated_matrix)
    # for i in range(0, 28):
    #     for j in range(0, 28):
    #         new_location = rotate_pixel(np.array([i,j]), rotation)
    #         if new_location[0] < 0:
    #             new_location[0] = 0.
    #         if (new_location[0] > 0 and new_location[0] < 27) and (new_location[1] > 0 or new_location[1] < 27):
    #             rotated_matrix[new_location[0]][new_location[1]] = matrix[i][j] 
    rotated_matrix = rotate(matrix, degree, reshape=False).astype(int)
    return rotated_matrix

#takes a pixel and rotates it by a degree
#def rotate_pixel(location, rotation):
    #print(location)
    #rotated_pixel = np.array([0,0])
    #rotated_pixel[0] = location[0]*rotation[0][0] + location[1]*rotation[0][1].astype(int)
    #rotated_pixel[1] = location[0]*rotation[1][0] + location[1]*rotation[1][1].astype(int)
    #rotated_pixel = np.dot(rotation, location)
    #rotated_pixel = np.matmul(rotation, location)
    #print(location, rotated_pixel.astype(int))
    #return rotated_pixel

# #def test_rotate_image():
#      # make a matrix
#      matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#      # make a function that rotates the matrix by 45 degrees
#      rotated_matrix = rotate_image(matrix, 90)
#      # make a matrix that is the expected result
#      expected_matrix = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])
#      # compare the expected matrix with the rotated matrix
#      for i in range(0, 3):
#         for j in range(0, 3):
#             assert rotated_matrix[i][j] == expected_matrix[i][j]
#if __name__ == '__main__':
#    main()