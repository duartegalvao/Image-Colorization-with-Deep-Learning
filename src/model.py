import numpy as np
import tensorflow as tf
import datetime

class Model:

    def __init__(self):

        self.learning_rate = 0.001
        self.num_epochs = 200
        self.batch_size = 100

        self.compiled = False

        self.save_interval = 50

    def _MLP(self, X):

        a1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)(X)
        logits = tf.keras.layers.Dense(10)(a1)

        return logits
    

    def compile(self):

        if self.compiled:
            print('Model already compiled.')
            return
        self.compiled = True

        self.X = tf.placeholder(tf.float32, shape=(None, 3072), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, 10), name='Y')

        logits = self._MLP(self.X)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=logits))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Tensorboard.
        tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()

    def train(self, X_train, Y_train, X_val, Y_val):

        if not self.compiled:
            print('Compile model first.')
            return

        (m, n_X) = X_train.shape
        num_batches = int(m / self.batch_size)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Tensorboard.
            merged = tf.summary.merge_all()
            date = str(datetime.datetime.now()).replace(" ", "_")[:19]
            train_writer = tf.summary.FileWriter('logs/' + date + '/train', sess.graph)
            val_writer = tf.summary.FileWriter('logs/' + date + '/val')

            try:
                for epoch in range(self.num_epochs):

                    for b in range(num_batches):

                        start = b * self.batch_size
                        end   = min(b * self.batch_size + self.batch_size, m)
                        batch_x = X_train[start:end, :]
                        batch_y = Y_train[start:end, :]

                        sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y})

                    # Add training loss to log.
                    summary = sess.run(merged, feed_dict={self.X: batch_x, self.Y: batch_y})
                    train_writer.add_summary(summary, epoch)

                    # Add validation loss to log.
                    summary = sess.run(merged, feed_dict={self.X: X_val, self.Y: Y_val})
                    val_writer.add_summary(summary, epoch)

                    # Save model.
                    if epoch % self.save_interval == 0:
                        self.saver.save(sess, 'models/' + date + '/model', global_step=epoch)

            except KeyboardInterrupt:
                print("\nInterrupted")


    """def evaluate(self, X, Y):

        if not self.compiled:
            print('Compile model first.')
            return
    """
