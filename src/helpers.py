import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

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
        y = data[b'labels']

        # Pre-process.
        X = X / 255
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X -= mean_X
        X /= std_X

        y = np.array(y)
        Y = one_hot_encode(y, 10)

    return X, Y, y


def load_CIFAR(seed):
    X, Y, _ = load_batch('data_batch_1')

    for i in range(4):
        data_X, data_Y, _ = load_batch('data_batch_{0}'.format(i+2))

        X = np.vstack((X, data_X))
        Y = np.vstack((Y, data_Y))
    
    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])

    X = X[p,:]
    Y = Y[p,:]

    X_train = X[5000:,:]
    Y_train = Y[5000:,:]

    X_val = X[:5000,:]
    Y_val = Y[:5000,:]

    X_test, Y_test, _ = load_batch('test_batch')

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

