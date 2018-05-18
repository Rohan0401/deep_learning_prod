import os

from os.path import isfile, isdir
from urllib.request import urlretrieve

from scipy.io import loadmat
from dl_progress import DLProgress

data_dir = 'data/'

def download_train_test_data():

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    if not isfile (data_dir + "train_32x32.mat"):
        with DLProgress (unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve (
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + 'train_32x32.mat',
                pbar.hook)

    if not isfile (data_dir + "test_32x32.mat"):
        with DLProgress (unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve (
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + 'test_32x32.mat',
                pbar.hook)



def load_data_sets():
    '''
    Loads train and test sets from the Matlab format
    :return: Dictionaries of the train and test datasets. Keys are
             'X' or 'y', image shape is (width, height, channels, dataset_size)
    '''
    if not os.path.exists(data_dir):
        raise Exception("Data directory doesn't exist!")

    train_set = loadmat(data_dir + 'train_32x32.mat')
    test_set = loadmat(data_dir + 'test_32x32.mat')
    return train_set, test_set

def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min ()) / (255 - x.min ()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x