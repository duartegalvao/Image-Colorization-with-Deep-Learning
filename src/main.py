import numpy as np
import tensorflow as tf

from model import Model

from helpers import *

SEED = 24

X_train, X_val, X_test = load_CIFAR(SEED)

np.random.seed(SEED)
tf.random.set_random_seed(SEED)


with tf.Session() as sess:
    
    MLP = Model(sess, SEED)
    
    MLP.compile()
    
    MLP.train(X_train, Y_train, X_val, Y_val)
    
    """print(MLP.evaluate(X_train, Y_train))
    print(MLP.evaluate(X_val, Y_val))
    print(MLP.evaluate(X_test, Y_test))
    """
    