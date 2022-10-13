# dont know if it works
# import the Python Image processing Library
from PIL import Image   

# make a function that takes image at a time and sends it to the function that rotates it, it gets a number of images, it should return all the rotated images
def rotate_image(image):
    rotated_images = []
    for i in range(0, len(image)):
        rotated_images[i] = image_rotation(image[i])
    return rotated_images
# make a function that rotates the image by 45 degrees and saves it in a new image, it gets one image to rotate and returns the new images
def image_rotation(image):
    rotated_image45 = image.rotate(45)
    rotated_image90 = image.rotate(90)
    rotated_image135 = image.rotate(135)
    rotated_image180 = image.rotate(180)
    images = [image, rotated_image45, rotated_image90, rotated_image135, rotated_image180]
    return images