import os

import numpy as np
from six.moves import cPickle as pickle
from skimage import color, io
from pathlib import Path


DATA_PATH = Path("data")

def one_hot_encode(y, num_classes=10):
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
        y = data[b'labels']

        y = np.array(y)
        Y = one_hot_encode(y, 10)

    return X, Y


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


def _split_luminosity_and_lab(rgb_images):
    lab = color.rgb2lab(rgb_images)
    lab_scaled = (lab + [-50., 0.5, 0.5]) / [50., 127.5, 127.5]

    X = lab_scaled[:, :, :, 0:1]
    Y = lab_scaled
    return X, Y


def _calculate_hue(rgb_images):
    hues = np.zeros(rgb_images.shape[0])
    for i in range(rgb_images.shape[0]):
        hues[i] = np.mean(color.rgb2hsv(rgb_images[i])[:,:,0])
    return hues


def _split_data_evenly(data, hue, ratio):
    inv_ratio = np.round(1./ratio).astype(np.int)
    hue_order_idx = np.argsort(hue)
    data_sorted = data[hue_order_idx,:,:,:]

    #_DEBUG_save_rgb_images(data_sorted, filename="images/test_grayscale/{}.png")
    # Delete B&W images from CIFAR
    data_sorted = data_sorted[582:,:,:,:]

    val_samples_idx = np.arange(0, data_sorted.shape[0], inv_ratio)
    data_train = np.delete(data_sorted, val_samples_idx, axis=0)
    data_val = data_sorted[val_samples_idx,:,:,:]
    return data_train, data_val


def _preproc_CIFAR(seed, ratio, smart_split):
    data, labels = _load_batch('data_batch_1')

    for i in range(4):
        data_batch, labels_batch = _load_batch('data_batch_{0}'.format(i + 2))

        data = np.vstack((data, data_batch))
        labels =  np.vstack((labels, labels_batch))

    data_test, labels_test = _load_batch('test_batch')

    data = data.reshape((data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    data_test = data_test.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    np.random.seed(seed)

    if smart_split:
        hue = _calculate_hue(data)
        data_train, data_val = _split_data_evenly(data, hue, ratio)
        
        p_train = np.random.permutation(data_train.shape[0])
        p_val = np.random.permutation(data_val.shape[0])
        data_train = data_train[p_train,:,:,:]
        data_val = data_val[p_val,:,:,:]
    else:
        p = np.random.permutation(data.shape[0])
        data = data[p,:,:,:]
        labels = labels[p,:]
        n_val = np.round(ratio * data.shape[0]).astype(np.int)
        data_train = data[n_val:,:,:,:]
        data_val = data[:n_val,:,:,:]

        labels_train = labels[n_val:,:]
        labels_val = labels[:n_val,:]

    X_train, Y_train = _split_luminosity_and_lab(data_train)
    X_val, Y_val = _split_luminosity_and_lab(data_val)
    X_test, Y_test = _split_luminosity_and_lab(data_test)

    return X_train, Y_train, labels_train, X_val, Y_val, labels_val, X_test, Y_test, labels_test


def load_CIFAR(seed, force=False, ratio=0.1, smart_split=True):

    cifar = _preproc_CIFAR(seed, ratio, smart_split)
    #_save_preprocd_CIFAR(cifar, seed)
    return cifar


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


def _DEBUG_save_rgb_images(img_batch, filename="images/output_{}.png"):
    for i in range(img_batch.shape[0]):
        filename_i = filename.format(i)
        directory = os.path.dirname(filename_i)
        if not os.path.exists(directory):
            os.makedirs(directory)

        img_save = np.round(img_batch[i]).astype(np.uint8)
        io.imsave(filename_i, img_save)
