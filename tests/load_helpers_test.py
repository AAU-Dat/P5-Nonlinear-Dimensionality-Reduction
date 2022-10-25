import src.load_helpers
import src.load_mnist
import os
import git


def test_download_files():
    file_names = ['train_file_image', 'train_file_label', 'test_file_image', 'test_file_label']
    directory = "mnist_data_test"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory
    
    src.load_helpers.create_folder(path)
    src.load_helpers.download_files(file_names, path)
    # assert that each file in file_names exists in path
    for file_name in file_names:
        assert os.path.exists(path + "/" + file_name)

def test_load_mnist_image():
    file_name = 'train_file_image'
    directory = "mnist_data_test"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory
    number_of_images = 60000
    image_data = src.load_helpers.load_mnist_image(path, file_name, number_of_images)
    assert len(image_data) == 47040000

def test_load_mnist_label():
    file_name = 'train_file_label'
    directory = "mnist_data_test"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory
    number_of_labels = 60000
    label_data = src.load_helpers.load_mnist_label(path, file_name, number_of_labels)
    assert len(label_data) == 60000

def test_load_mnist_all():
    file_names = ['train_file_image', 'train_file_label', 'test_file_image', 'test_file_label']
    directory = "mnist_data_test"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory
    number_of_images = 60000
    (train_data, test_data) = src.load_helpers.load_mnist_all(path, file_names, number_of_images)
    assert len(train_data[0]) == 47040000
    assert len(train_data[1]) == 60000
    assert len(test_data[0]) == 7840000
    assert len(test_data[1]) == 10000

def test_reshape_data():
    file_names = ['train_file_image', 'train_file_label', 'test_file_image', 'test_file_label']
    directory = "mnist_data_test"
    repo = git.Repo('.', search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
    path = git_root + "/src/" + directory
    number_of_images = 60000
    (train_data, test_data) = src.load_helpers.load_mnist_all(path, file_names, number_of_images)
    (train_data, test_data) = src.load_helpers.reshape_data((train_data, test_data))
    assert train_data[0].shape == (60000, 784)
    assert test_data[0].shape == (10000, 784)