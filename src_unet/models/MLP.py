import tensorflow as tf

class MLP:

    def __init__(self, X, seed):
        self.X = X
        self.seed = seed

        self.initializer = tf.glorot_uniform_initializer(self.seed)

    def forward(self):

        a1 = tf.keras.layers.Dense(20,
                                activation=tf.nn.relu,
                                kernel_initializer=self.initializer)(self.X)
        logits = tf.keras.layers.Dense(10,
                                kernel_initializer=self.initializer)(a1)

        return logits

        