"""
from keras.datasets import mnist
from matplotlib import pyplot
import pylab

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
 

for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))

print("yes")

"""
import matplotlib.pyplot as plt

# Define Data

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plot

plt.plot(x, y)

# Display

plt.show()