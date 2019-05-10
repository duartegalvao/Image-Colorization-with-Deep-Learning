import os

import numpy as np
from six.moves import cPickle as pickle
from skimage import color, io
from pathlib import Path


DATA_PATH = Path("data")


def _load_batch(file_path):
    """
        Loads data given path to file.

        Args:
            - file_path: path to file.

        Return:
            - X: X data.
            - Y: One hot encoding.
            - y: labels.
    """
    with open(str(DATA_PATH / file_path), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X = data[b'data']
    return X


def _is_CIFAR_preprocd(seed):
    return (DATA_PATH / "preproc_{}.npz".format(seed)).exists()


def _load_preprocd_CIFAR(seed):
    with np.load(DATA_PATH / "preproc_{}.npz".format(seed)) as data:
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_val = data['X_val']
        Y_val = data['Y_val']
        X_test = data['X_test']
        Y_test = data['Y_test']
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def _save_preprocd_CIFAR(cifar, seed):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = cifar
    np.savez(DATA_PATH / "preproc_{}.npz".format(seed),
             X_train=X_train,
             Y_train=Y_train,
             X_val=X_val,
             Y_val=Y_val,
             X_test=X_test,
             Y_test=Y_test)


def _preproc_CIFAR(seed):
    data = _load_batch('data_batch_1')

    for i in range(4):
        data_batch = _load_batch('data_batch_{0}'.format(i + 2))

        data = np.vstack((data, data_batch))
    
    np.random.seed(seed)
    p = np.random.permutation(data.shape[0])

    data = data[p, :]

    data_train = data[5000:, :]
    data_val = data[:5000, :]
    data_test = _load_batch('test_batch')

    data_train = data_train.reshape((45000, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_val = data_val.reshape((5000, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_test = data_test.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    X_train, Y_train = split_luminosity_and_lab(data_train)
    X_val, Y_val = split_luminosity_and_lab(data_val)
    X_test, Y_test = split_luminosity_and_lab(data_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_CIFAR(seed):
    if _is_CIFAR_preprocd(seed):
        print("Found preprocessed data for seed={}.".format(seed))
        return _load_preprocd_CIFAR(seed)
    else:
        print("Found no preprocessed data. Pre-processing for seed={}.".format(seed))
        cifar = _preproc_CIFAR(seed)
        _save_preprocd_CIFAR(cifar, seed)
        return cifar


def split_luminosity_and_lab(rgb_images):
    lab = color.rgb2lab(rgb_images)
    lab_scaled = (lab + [-50., 0.5, 0.5]) / [50., 127.5, 127.5]

    X = lab_scaled[:, :, :, 0:1]
    Y = lab_scaled
    return X, Y


def save_lab_images(img_batch, filename="images/output_{}.png"):
    lab_unscaled = (img_batch * [50., 127.5, 127.5]) - [-50., 0.5, 0.5]

    for i in range(lab_unscaled.shape[0]):
        filename_i = filename.format(i)
        directory = os.path.dirname(filename_i)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        rgb = color.lab2rgb(lab_unscaled[i])
        img_save = np.round(rgb*255).astype(np.uint8)
        io.imsave(filename.format(i), img_save)


def save_gray_images(img_batch, filename="images/output_{}.png"):
    for i in range(img_batch.shape[0]):
        filename_i = filename.format(i)
        directory = os.path.dirname(filename_i)
        if not os.path.exists(directory):
            os.makedirs(directory)

        img_save = np.round(img_batch[i] * 255).astype(np.uint8)
        io.imsave(filename_i, img_save)
