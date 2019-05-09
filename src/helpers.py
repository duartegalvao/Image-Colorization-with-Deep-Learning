import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from skimage import color


def one_hot_encode(y, num_classes):
    """
        One-hot encoding.

        Args:
            - y : vector to encode.
            - num_classes: number of classes.
    """
    N = y.shape[0]

    encoded = np.zeros((N, num_classes))
    
    for i, label in enumerate(y):
        encoded[i][label] = 1
    
    return encoded


def load_batch(file_path):
    """
        Loads data given path to file.

        Args:
            - file_path: path to file.

        Return:
            - X: X data.
            - Y: One hot encoding.
            - y: labels.
    """
    path = "data/" + file_path

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X = data[b'data']

        # Pre-process.
        X = X / 255
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X -= mean_X
        X /= std_X

    return X


def load_CIFAR(seed):
    data = load_batch('data_batch_1')

    for i in range(4):
        data_batch = load_batch('data_batch_{0}'.format(i+2))

        data = np.vstack((data, data_batch))
    
    np.random.seed(seed)
    p = np.random.permutation(data.shape[0])

    data = data[p, :]

    data_train = data[5000:, :]
    data_val = data[:5000, :]
    data_test = load_batch('test_batch')

    data_train = data_train.reshape((45000, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_val = data_val.reshape((5000, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_test = data_test.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    X_train, Y_train = split_luminosity_and_lab(data_train)
    X_val, Y_val = split_luminosity_and_lab(data_val)
    X_test, Y_test = split_luminosity_and_lab(data_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def split_luminosity_and_lab(rgb_images):
    lab = color.rgb2lab(rgb_images)
    lab_scaled = (lab + [-50., 0.5, 0.5]) / [50., 127.5, 127.5]

    X = lab_scaled[:, :, :, 0:1]
    Y = lab_scaled
    return X, Y
