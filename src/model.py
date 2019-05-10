import os
import numpy as np
import tensorflow as tf
import datetime

from models.Generator import Generator
from models.Discriminator import Discriminator

class Model:

    def __init__(self, sess, seed):

        # TODO: put these parameters as arguments or something.

        # Training settings.
        self.learning_rate = 0.001
        self.num_epochs = 200
        self.batch_size = 128
        self.shuffle = False

        self.compiled = False

        # Verbose/logs/checkpoints options.
        self.verbose = True
        self.log = True
        self.save = True
        self.save_interval = 20

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

        self.labels = tf.placeholder(tf.float32, shape=(None, 1), name='labels')

        # Generator.
        generator = Generator(self.seed)
        gen_out = generator.forward(self.X)

        # Discriminator.
        discriminator = Discriminator(self.seed)

        disc_out_fake = discriminator.forward(tf.concat([self.X, gen_out], 3))
        disc_out_real = discriminator.forward(tf.concat([self.X, self.Y], 3), reuse_vars=True)

        # Generator loss.
        # TODO: add l1 loss.
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_fake, labels=tf.ones_like(disc_out_fake)))

        # Discriminator loss.
        self.disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_fake, labels=tf.zeros_like(disc_out_fake)))

        # TODO: smoothing
        self.disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_real, labels=tf.ones_like(disc_out_real)))

        self.disc_loss = tf.reduce_mean(self.disc_loss_fake + self.disc_loss_real)

        # Optimizer.
        self.gen_optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self.learning_rate
                                    ).minimize(self.gen_loss, var_list=generator.variables)

        self.disc_optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self.learning_rate
                                    ).minimize(self.disc_loss, var_list=discriminator.variables)
        
        # Tensorboard.
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('disc_loss', self.disc_loss)

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
                epoch_gen_loss = 0.0
                epoch_disc_loss = 0.0

                if self.shuffle:
                    p = np.random.permutation(X_train.shape[0])
                    X_train = X_train[p,:,:,:]
                    Y_train = Y_train[p,:,:,:]

                for b in range(num_batches):

                    start = b * self.batch_size
                    end   = min(b * self.batch_size + self.batch_size, N)
                    batch_x = X_train[start:end,:,:,:]
                    batch_y = Y_train[start:end,:,:,:]

                    _, l_disc = self.sess.run([self.disc_optimizer, self.disc_loss], feed_dict={self.X: batch_x ,self.Y: batch_y})

                    _, l_gen = self.sess.run([self.gen_optimizer, self.gen_loss], feed_dict={self.X: batch_x})
                    _, l_gen = self.sess.run([self.gen_optimizer, self.gen_loss], feed_dict={self.X: batch_x})

                    epoch_gen_loss += l_gen / num_batches
                    epoch_disc_loss += l_disc / num_batches

                if self.verbose:
                    print('Epoch:', (epoch+1), 'gen_loss =', epoch_gen_loss)
                    print('Epoch:', (epoch+1), 'disc_loss =', epoch_disc_loss)

                if self.log:
                    # Add training epoch loss to train log.
                    summary = tf.Summary()
                    summary.value.add(tag='gen_loss', simple_value=epoch_gen_loss)
                    summary.value.add(tag='disc_loss', simple_value=epoch_disc_loss)
                    train_writer.add_summary(summary, epoch)
                    train_writer.flush()

                    # Add validation loss to val log.
                    summary = self.sess.run(merged, feed_dict={self.X: X_val ,self.Y: Y_val})
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
