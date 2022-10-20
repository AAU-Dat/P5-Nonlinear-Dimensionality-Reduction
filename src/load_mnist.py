import load_helpers as load
import git

# Load the MNIST data and return a tuple containing the training data and the test data with the images and labels
def load_mnist(sample_size = 60000):
    directory = "mnist_data"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory

    file_names = ['train_file_image', 'train_file_label', 'test_file_image', 'test_file_label']

    load.create_folder(path)
    load.check_for_files(file_names, path)
    return load.reshape_data(load.load_mnist_all(path, file_names, sample_size))