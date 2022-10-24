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

