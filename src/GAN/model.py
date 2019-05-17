import os
import numpy as np
import tensorflow as tf
import datetime

from models.Generator import Generator
from models.Discriminator import Discriminator

from helpers import save_lab_images

class Model:
    """
        GAN.
    """

    def __init__(self, sess, seed):

        self.compiled = False

        # Training settings.
        self.num_epochs = 300
        self.batch_size = 128
        self.shuffle = True
        self.learning_rate = 0.0003
        self.learning_rate_decay = True
        self.learning_rate_decay_steps = 20000.0
        self.learning_rate_decay_rate = 0.1

        # Verbose/logs/checkpoints options.
        self.verbose = True
        self.log = True
        self.save = True
        self.save_interval = 10
        self.validate = True
        self.sample_interval = 10
        self.num_samples = 20

        # GAN parameters.
        self.label_smoothing = 1.0
        self.l1_weight = 100.0

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

        # Generator.
        generator = Generator(self.seed)

        # Discriminator.
        discriminator = Discriminator(self.seed)

        self.gen_out = generator.forward(self.X)
        disc_out_real = discriminator.forward(tf.concat([self.X, self.Y], 3))
        disc_out_fake = discriminator.forward(tf.concat([self.X, self.gen_out], 3), reuse_vars=True)

        # Generator loss.
        self.gen_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_fake, labels=tf.ones_like(disc_out_fake)))
        self.gen_loss_l1 = tf.reduce_mean(tf.abs(self.Y - self.gen_out)) * self.l1_weight
        self.gen_loss = self.gen_loss_gan + self.gen_loss_l1

        # Discriminator losses.
        disc_l_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_fake, labels=tf.zeros_like(disc_out_fake))
        disc_l_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out_real, labels=tf.ones_like(disc_out_real)*self.label_smoothing)
        self.disc_loss_fake = tf.reduce_mean(disc_l_fake)
        self.disc_loss_real = tf.reduce_mean(disc_l_real)
        self.disc_loss = tf.reduce_mean(disc_l_fake + disc_l_real)

        # Global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Learning rate.
        if self.learning_rate_decay:
            self.lr = tf.maximum(1e-6, tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=self.global_step,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay_rate))
        else:
            self.lr = tf.constant(self.learning_rate)

        # Optimizers.
        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gen_loss, var_list=generator.variables)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10).minimize(self.disc_loss, var_list=discriminator.variables, global_step=self.global_step)

        # Sampler.
        gen_sample = Generator(self.seed, is_training=False)
        self.sampler = gen_sample.forward(self.X, reuse_vars=True)
        self.MAE = tf.reduce_mean(tf.abs(self.Y - self.sampler))

        self.saver = tf.train.Saver()


    def train(self, X_train, Y_train, X_val, Y_val):

        if not self.compiled:
            print('Compile model first.')
            return

        N = X_train.shape[0]
        num_batches = int(N / self.batch_size)
    
        # Tensorboard.
        date = str(datetime.datetime.now()).replace(" ", "_")[:19]
        if not os.path.exists('checkpoints/' + date):
            os.makedirs('checkpoints/' + date)
        train_writer = tf.summary.FileWriter('logs/' + date + '/train', self.sess.graph)
        train_writer.flush()

        self.sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(self.num_epochs):
                epoch_gen_loss = 0.0
                epoch_gen_gan_loss = 0.0
                epoch_gen_l1_loss = 0.0
                epoch_disc_loss = 0.0
                epoch_disc_real_loss = 0.0
                epoch_disc_fake_loss = 0.0

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

                    _, l_disc, l_disc_fake, l_disc_real = self.sess.run([self.disc_optimizer, self.disc_loss, self.disc_loss_fake, self.disc_loss_real], feed_dict=feed)

                    _, l_gen, l_gen_gan, l_gen_l1 = self.sess.run([self.gen_optimizer, self.gen_loss, self.gen_loss_gan, self.gen_loss_l1], feed_dict=feed)
                    _, l_gen, l_gen_gan, l_gen_l1 = self.sess.run([self.gen_optimizer, self.gen_loss, self.gen_loss_gan, self.gen_loss_l1], feed_dict=feed)

                    epoch_gen_loss += l_gen / num_batches
                    epoch_gen_gan_loss += l_gen_gan / num_batches
                    epoch_gen_l1_loss += l_gen_l1 / num_batches
                    epoch_disc_loss += l_disc / num_batches
                    epoch_disc_fake_loss += l_disc_fake / num_batches
                    epoch_disc_real_loss += l_disc_real / num_batches

                lr = self.sess.run(self.lr)

                if self.verbose:
                    print('Epoch: {0}'.format(epoch+1))
                    print('learning_rate =', lr)
                    print('gen_loss =', epoch_gen_loss)
                    print('gen_gan_loss =', epoch_gen_gan_loss)
                    print('gen_l1_loss =', epoch_gen_l1_loss)
                    print('disc_loss =', epoch_disc_loss)
                    print('disc_fake_loss =', epoch_disc_fake_loss)
                    print('disc_real_loss =', epoch_disc_real_loss, ' \n')

                # Add training losses to train log (tensorboard).
                if self.log:
                    summary = tf.Summary()
                    summary.value.add(tag='learning_rate', simple_value=lr)
                    summary.value.add(tag='gen_loss', simple_value=epoch_gen_loss)
                    summary.value.add(tag='gen_gan_loss', simple_value=epoch_gen_gan_loss)
                    summary.value.add(tag='gen_l1_loss', simple_value=epoch_gen_l1_loss)
                    summary.value.add(tag='disc_loss', simple_value=epoch_disc_loss)
                    summary.value.add(tag='disc_fake_loss', simple_value=epoch_disc_fake_loss)
                    summary.value.add(tag='disc_real_loss', simple_value=epoch_disc_real_loss)
                    train_writer.add_summary(summary, epoch)
                    train_writer.flush()

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

            l = self.sess.run(self.MAE, feed_dict={self.X: batch_x, self.Y: batch_y})

            loss += l / num_batches

        return loss