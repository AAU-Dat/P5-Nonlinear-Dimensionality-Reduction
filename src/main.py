import load as mnistload


def main():
    mnistload.load_mnist_all("../mnist-original-dataset/train-images-idx3-ubyte.gz",
                             "../mnist-original-dataset/train-labels-idx1-ubyte.gz",
                             "../mnist-original-dataset/t10k-images-idx3-ubyte.gz",
                             "../mnist-original-dataset/t10k-labels-idx1-ubyte.gz")
