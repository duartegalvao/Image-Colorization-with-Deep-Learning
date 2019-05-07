import numpy as np
import tensorflow as tf

class Model:

    def __init__(self):

        self.learning_rate = 0.01
        self.num_epochs = 200
        self.batch_size = 100

        self.compiled = False


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


    def train(self, X_train, Y_train):

        if not self.compiled:
            print('Compile model first.')
            return

        (m, n_X) = X_train.shape
        num_batch = int(m / self.batch_size)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                for epoch in range(self.num_epochs):
                    epoch_cost = 0.0

                    for i in range(num_batch):
                        start = i * self.batch_size
                        end   = min(i * self.batch_size + self.batch_size, m)
                        batch_x = X_train[start:end, :]
                        batch_y = Y_train[start:end, :]

                        _, c = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})

                        epoch_cost += c / num_batch

                    print('Epoch:', (epoch +1), 'cost =', epoch_cost)

                    # Calculate error on validation set.
                    #if isinstance(X_val, np.ndarray):
                    #    cost = sess.run(mse, feed_dict={X: X_val})
                    #    val_costs.append(cost)

            except KeyboardInterrupt:
                print("\nInterrupted!")


    """def evaluate(self, X, Y):

        if not self.compiled:
            print('Compile model first.')
            return
    """
