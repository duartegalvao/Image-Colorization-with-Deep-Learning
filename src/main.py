import numpy as np
import tensorflow as tf

from model import Model

from helpers import *

SEED = 24

X_train, X_val, _ = load_CIFAR(SEED)

X_train = X_train[0:1000,:,:,:]
X_val = X_val[0:50,:,:,:]

print(X_train.shape)
print(X_val.shape)

np.random.seed(SEED)
tf.random.set_random_seed(SEED)


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
    UNET = Model(sess, SEED)
    
    UNET.compile()
    
    UNET.train(X_train, X_val)

    UNET.predict(X_train)
    
    """print(MLP.evaluate(X_train, Y_train))
    print(MLP.evaluate(X_val, Y_val))
    print(MLP.evaluate(X_test, Y_test))
    """
    