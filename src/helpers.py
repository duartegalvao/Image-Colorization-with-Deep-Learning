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

    encoded = np.zeros((num_classes, N))
    
    for i, label in enumerate(y):
        encoded[label][i] = 1
    
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
    path = "data/cifar-10-batches-py/" + file_path

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X = data[b'data']
        y = data[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose([2,3,1,0])
        X = X.reshape(3072, 10000)

        # Pre-process.
        X = X / 255
        mean_X = np.mean(X, axis=1, keepdims=True)
        std_X = np.std(X, axis=1, keepdims=True)
        X -= mean_X
        X /= std_X

        y = np.array(y)
        Y = one_hot_encode(y, 10)

    return X, Y, y


def plot_images(data, labels):
    """
        Plots images in grid.
    """
    fig = plt.figure(figsize=(16, 10))

    for i in range(28):
        ax = fig.add_subplot(4, 7, i+1)
        ax.imshow(data[:,:,:,i], origin='upper')
        ax.set_title('Class {0}'.format(labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()


def parameter_grad(model, parameter, X, Y, h=1e-5):
    dParam = np.empty_like(parameter)

    it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        old_val = parameter[ix]

        parameter[ix] = old_val + h
        c1, _ = model.evaluate(X, Y)

        parameter[ix] = old_val - h
        c2, _ = model.evaluate(X, Y)

        parameter[ix] = old_val

        # Compute gradient.
        dParam[ix] = (c1 - c2) / (2 * h)

        it.iternext()

    return dParam



def compute_numerical_grad_network(model, X, Y, h=1e-5):
    """
        Numerically compute gradients (precise).

        Args:
            - model: model instance.
            - X, Y
            - h: infinitesimal
    """
    grads = {}
    i = 1
    j = 1

    for layer in model.layers:
        if layer.name == 'Dense':
            dW = parameter_grad(model, layer.W, X, Y)
            grads['W{0}'.format(i)] = dW
            db = parameter_grad(model, layer.b, X, Y)
            grads['b{0}'.format(i)] = db
            i+=1
        elif layer.name == 'BatchNorm':
            dbeta = parameter_grad(model, layer.beta, X, Y)
            grads['beta{0}'.format(j)] = dbeta
            dgamma = parameter_grad(model, layer.gamma, X, Y)
            grads['gamma{0}'.format(j)] = dgamma
            j+=1

    return grads

def compute_numerical_grad(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()

    return grad


def compute_numerical_grad_fast(f, X, Y, parameters, h=1e-5):
    """
        Numerically compute gradients (less precise).

        Args:
            - f : function to compute gradient at.
            - X, Y
            - parameters: model parameters.
            - h: infinitesimal
    """
    grads = {}

    c = f(X, Y, parameters)

    for p in parameters.keys():

        dParam = np.zeros_like(parameters[p])

        it = np.nditer(parameters[p], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:

            ix = it.multi_index

            old_val = parameters[p][ix]

            parameters[p][ix] = old_val + h
            c1 = f(X, Y, parameters)

            parameters[p][ix] = old_val

            # Compute gradient.
            dParam[ix] = (c1 - c) / h

            it.iternext()
    
        grads[p] = dParam

    return grads
