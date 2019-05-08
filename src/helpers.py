import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
#from skimage import color


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
    X = load_batch('data_batch_1')

    for i in range(4):
        data_X = load_batch('data_batch_{0}'.format(i+2))

        X = np.vstack((X, data_X))
    
    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])

    X = X[p,:]

    X_train = X[5000:,:]

    X_val = X[:5000,:]

    X_test = load_batch('test_batch')

    X_train = X_train.reshape((45000, 3, 32, 32)).transpose(0, 2, 3, 1)
    X_val = X_val.reshape((5000, 3, 32, 32)).transpose(0, 2, 3, 1)
    X_test = X_test.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    return X_train, X_val, X_test


"""def split_luminosity_and_lab(rgb_images):
    lab = color.rgb2lab(rgb_images)
    lab_scaled = (lab + [-50., 0.5, 0.5]) / [50., 127.5, 127.5]

    X = lab_scaled[:, :, :, 0]
    Y = lab_scaled
    return X, Y
"""