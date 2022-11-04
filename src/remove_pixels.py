import numpy as np
from numpy.typing import ArrayLike
import load_mnist as lm
import matplotlib.pyplot as plt

#this function is the wrapper function for the remove pixel augmentation. It creates new np arrays so ist possible to write data directly to the array.
def remove_pixel_augmentation(data: ArrayLike, size) -> ArrayLike:
    training_pictures = data[0][0]
    test_pictures = data[1][0]
    new_training_pictures = np.array(training_pictures, copy=True)
    new_test_pictures = np.array(test_pictures, copy=True)
    for picture in new_test_pictures:
        picture = find_pixel_cluster(picture, size)

    for picture in new_training_pictures:
        picture = find_pixel_cluster(picture, size)

    return create_new_tuple_from_augment(
        data,
        new_training_pictures=new_training_pictures,
        new_test_pictures=new_test_pictures,
    )

#this function essentially takes newly poccessed data combines it with the old data and returns a new tuple.
def create_new_tuple_from_augment(
    data: ArrayLike, new_training_pictures: ArrayLike, new_test_pictures: ArrayLike
):
    training_data_tuple = (
        np.concatenate((data[0][0], new_training_pictures), axis=0),
        np.concatenate((data[0][1], data[0][1]), axis=None),
    )
    test_data_tuple = (
        np.concatenate((data[1][0], new_test_pictures), axis=0),
        np.concatenate((data[1][1], data[1][1]), axis=None),
    )

    return (training_data_tuple, test_data_tuple)

#this function iterates throuh a picture to find a valid cluster to remove. a valid cluster is a cluster of pixels where none are white.
def find_pixel_cluster(picture: ArrayLike, size):

    # np.nonzero creates a tuple value so if we want the array we have to take the first element in the tuple.
    # this we assign to black_indixes because thats what we want to work on.
    black_indixes = np.nonzero(picture)
    black_indixes = black_indixes[0]
    np.random.shuffle(black_indixes)
    for index in black_indixes:
        valid = True
        if index + (size - 1) * 28 <= 784:
            cluster = create_pixel_cluster_28x28(index, size)
            for pixel in cluster:
                if picture[pixel] == 0:
                    valid = False
                    break

            if valid:
                remove_pixel_cluster(picture, cluster)
                return picture

    return picture

#this function takes a pixel position and creates an array of all other relevant pixel for a cluster of the given size.
def create_pixel_cluster_28x28(index, size):
    array = []
    for i in range(0, size):
        for j in range(0, size):
            array.append(index + ((i * 28) + (j)))

    return array

#takes a picture and removes pixel based on a array of indexes 
def remove_pixel_cluster(picture: ArrayLike, cluster):
    for pixel in cluster:
        picture[pixel] = 0


def plot_digits(data):
    fig, axes = plt.subplots(
        4,
        10,
        figsize=(10, 4),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(28, 28),
            cmap="binary",
            interpolation="nearest",
            clim=(0, 16),
        )

    fig.savefig("hello4")


def test():
    data = lm.load_mnist()
    data = remove_pixel_augmentation(data, 3)
    training_pictures = data[0][0]
    print(data[0][1][:20])
    plot_digits(training_pictures[60000:])


def test_wrapper():
    data = lm.load_mnist()
    augmented_data = remove_pixel_augmentation(data, 3)
    print(augmented_data[0][0].shape)
    print(augmented_data[1][0].shape)
    print(data[0][0][0] == augmented_data[0][0][60000])


def test_cluster():
    index = 1
    print(create_pixel_cluster_28x28(index, 2))