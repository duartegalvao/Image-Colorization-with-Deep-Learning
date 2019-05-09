import numpy as np
import tensorflow as tf

from model import Model

from helpers import *

SEED = 24

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_CIFAR(SEED)

#X_train = X_train[0:1000,:,:,:]
#Y_train = Y_train[0:1000,:,:,:]

#X_val = X_val[0:100,:,:,:]
#Y_val = Y_val[0:100,:,:,:]

#X_test = X_test[0:100,:,:,:]
#Y_test = Y_test[0:100,:,:,:]

print('Train:')
print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)

print('Validation:')
print('X_val:', X_val.shape)
print('Y_val:', Y_val.shape)

print('Test:')
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)

#save_gray_images(X_train[0:10,:,:,:], filename="images/train_{}/before_gray.png")
save_lab_images(Y_train[0:20,:,:,:], filename="images/train_{}/before_color.png")

#save_gray_images(X_val[0:10,:,:,:], filename="images/val_{}/before_gray.png")
save_lab_images(Y_val[0:20,:,:,:], filename="images/val_{}/before_color.png")

#save_gray_images(X_test[0:10,:,:,:], filename="images/test_{}/before_gray.png")
save_lab_images(Y_test[0:20,:,:,:], filename="images/test_{}/before_color.png")

np.random.seed(SEED)
tf.random.set_random_seed(SEED)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
    UNET = Model(sess, SEED)

    UNET.compile()
    
    print('Training...')
    UNET.train(X_train, Y_train[:,:,:,1:3], X_val, Y_val[:,:,:,1:3])

    print('Predicting training set...')
    pred = UNET.predict(X_train)
    pred = np.concatenate([X_train,pred], axis=3)
    save_lab_images(pred[0:20,:,:,:], filename="images/train_{}/after.png")

    print('Predicting validation set...')
    pred = UNET.predict(X_val)
    pred = np.concatenate([X_val,pred], axis=3)
    save_lab_images(pred[0:20,:,:,:], filename="images/val_{}/after.png")

    print('Predicting test set...')
    pred = UNET.predict(X_test)
    pred = np.concatenate([X_test,pred], axis=3)
    save_lab_images(pred[0:20,:,:,:], filename="images/test_{}/after.png")
