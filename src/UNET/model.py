import os
import numpy as np
import tensorflow as tf
import datetime

from models.UNet import UNet

from helpers import save_lab_images

class Model:
    """
        U-Net.
    """
    def __init__(self, sess, seed):

        self.compiled = False

        # Training settings.
        self.num_epochs = 200
        self.batch_size = 128
        self.shuffle = True
        self.learning_rate = 0.001
        self.learning_rate_decay = True
        self.learning_rate_decay_steps = 50
        self.learning_rate_decay_rate = 0.3
        
        # Verbose/logs/checkpoints options.
        self.verbose = True
        self.log = True
        self.save = True
        self.save_interval = 20
        self.validate = True
        self.sample_interval = 10
        self.num_samples = 20

        self.sess = sess
        self.seed = seed

        np.random.seed(self.seed * 2)


    def compile(self):

        if self.compiled:
            print('Model already compiled.')
            return
        self.compiled = True

        # Placeholders.
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, 32, 32, 2), name='Y')

        # U-Net.
        net = UNet(self.seed)
        self.out = net.forward(self.X)

        # Loss and metrics.
        # TODO: try with MAE.
        self.loss = tf.keras.losses.MeanSquaredError()(self.Y, self.out)

        # Global step.
        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')

        # Learning rate.
        if self.learning_rate_decay:
            self.lr = tf.train.exponential_decay(
                self.learning_rate,
                self.global_step,
                self.learning_rate_decay_steps,
                self.learning_rate_decay_rate,
                name='learning_rate_decay')
        else:
            self.lr = tf.constant(self.learning_rate)

        # Optimizer.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

        # Sampler.
        gen_sample = UNet(self.seed, is_training=False)
        self.sampler = gen_sample.forward(self.X, reuse_vars=True)

        # Tensorboard.
        tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()


    def train(self, X_train, Y_train, X_val, Y_val):

        if not self.compiled:
            print('Compile model first.')
            return

        N = X_train.shape[0]
        num_batches = int(N / self.batch_size)

        # Tensorboard.
        merged = tf.summary.merge_all()
        date = str(datetime.datetime.now()).replace(" ", "_")[:19]
        if not os.path.exists('checkpoints/' + date):
            os.makedirs('checkpoints/' + date)
        train_writer = tf.summary.FileWriter('logs/' + date + '/train', self.sess.graph)
        val_writer = tf.summary.FileWriter('logs/' + date + '/val')
        train_writer.flush()
        val_writer.flush()

        self.sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(self.num_epochs):
                epoch_train_loss = 0.0

                if self.shuffle:
                    p = np.random.permutation(X_train.shape[0])
                    X_train = X_train[p,:,:,:]
                    Y_train = Y_train[p,:,:,:]

                for b in range(num_batches):

                    start = b * self.batch_size
                    end   = min(b * self.batch_size + self.batch_size, N)
                    batch_x = X_train[start:end,:,:,:]
                    batch_y = Y_train[start:end,:,:,:]

                    feed = {self.X: batch_x, self.Y: batch_y}

                    _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed)

                    epoch_train_loss += l / num_batches

                lr = self.sess.run(self.lr)

                if self.verbose:
                    print('Epoch: {0}'.format(epoch+1))
                    print('learning_rate =', lr)
                    print('train_loss =', epoch_train_loss)

                # Add training/validation epoch loss to train log.
                if self.log:
                    summary = tf.Summary()
                    summary.value.add(tag='loss', simple_value=epoch_train_loss)
                    train_writer.add_summary(summary, epoch)
                    train_writer.flush()
                    summary = self.sess.run(merged,  feed_dict={self.X: X_val ,self.Y: Y_val})
                    val_writer.add_summary(summary, epoch)
                    val_writer.flush()

                # Sample model (validate).
                if self.validate and (epoch+1) % self.sample_interval == 0:
                    X_pred = X_val[0:self.num_samples,:,:,:]
                    samples = self.sample(X_pred)
                    samples = np.concatenate([X_pred,samples], axis=3)
                    save_lab_images(samples, filename="images/val_{}/epoch_" + str(epoch+1) + ".png")

                # Save model.
                if self.save and (epoch+1) % self.save_interval == 0:
                    self.saver.save(self.sess, 'checkpoints/' + date + '/model', global_step=epoch, write_meta_graph=False)

        except KeyboardInterrupt:
            print("\nInterrupted")


    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('Loading model from path: {0}'.format(os.path.join(path, ckpt_name)))
        self.saver.restore(self.sess, os.path.join(path, ckpt_name))


    def sample(self, X):
        if not self.compiled:
            print('Compile model first.')
            return

        return self.sess.run(self.sampler, feed_dict={self.X: X})


    def get_loss(self, X, Y):
        if not self.compiled:
            print('Compile model first.')
            return

        N = X.shape[0]
        num_batches = int(N / self.batch_size)

        loss = 0.0

        for b in range(num_batches):
            start = b * self.batch_size
            end   = min(b * self.batch_size + self.batch_size, N)
            batch_x = X[start:end,:,:,:]
            batch_y = Y[start:end,:,:,:]

            l = self.sess.run(self.loss, feed_dict={self.X: batch_x, self.Y: batch_y})

            loss += l / num_batches

        return loss
