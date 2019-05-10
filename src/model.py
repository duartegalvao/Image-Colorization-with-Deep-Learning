import os
import numpy as np
import tensorflow as tf
import datetime

from models.UNet import UNet

class Model:

    def __init__(self, sess, seed):

        # TODO: put these parameters as arguments or something.

        # Training settings.
        self.learning_rate = 0.001
        self.num_epochs = 200
        self.batch_size = 128

        self.compiled = False

        # Verbose/logs/checkpoints options.
        self.verbose = True
        self.log = True
        self.save = True
        self.save_interval = 20

        self.sess = sess
        self.seed = seed


    def compile(self):

        if self.compiled:
            print('Model already compiled.')
            return
        self.compiled = True

        # Placeholders.
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, 32, 32, 2), name='Y')

        # Model.
        net = UNet(self.seed)
        self.out = net.forward(self.X)

        # Loss and metrics.
        self.loss = tf.keras.losses.MeanSquaredError()(self.Y, self.out)
        #self.loss = tf.reduce_sum(tf.square(self.out - self.Y))

        # Optimizer.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Tensorboard.
        tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()


    def train(self, X_train, Y_train, X_val, Y_val):

        if not self.compiled:
            print('Compile model first.')
            return

        N = X_train.shape[0]
        num_batches = int(N / self.batch_size)
        
        self.sess.run(tf.global_variables_initializer())

        # Tensorboard.
        merged = tf.summary.merge_all()
        date = str(datetime.datetime.now()).replace(" ", "_")[:19]

        if not os.path.exists('checkpoints/' + date):
            os.makedirs('checkpoints/' + date)

        train_writer = tf.summary.FileWriter('logs/' + date + '/train', self.sess.graph)
        val_writer = tf.summary.FileWriter('logs/' + date + '/val')
        train_writer.flush()
        val_writer.flush()

        try:
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0

                for b in range(num_batches):

                    start = b * self.batch_size
                    end   = min(b * self.batch_size + self.batch_size, N)
                    batch_x = X_train[start:end,:,:,:]
                    batch_y = Y_train[start:end,:,:,:]

                    _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x ,self.Y: batch_y})

                    epoch_loss += l / num_batches

                if self.verbose:
                    print('Epoch:', (epoch+1), 'loss =', epoch_loss)

                if self.log:
                    # Add training epoch loss to train log.
                    summary = tf.Summary()
                    summary.value.add(tag='loss', simple_value=epoch_loss)
                    train_writer.add_summary(summary, epoch)
                    train_writer.flush()

                    # Add validation loss to val log.
                    summary = self.sess.run(merged,  feed_dict={self.X: X_val ,self.Y: Y_val})
                    val_writer.add_summary(summary, epoch)
                    val_writer.flush()

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


    def predict(self, X):

        if not self.compiled:
            print('Compile model first.')
            return

        return self.sess.run(self.out, feed_dict={self.X: X})
