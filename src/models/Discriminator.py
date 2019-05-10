import tensorflow as tf

class Discriminator:

    def __init__(self, seed):
        self.seed = seed
        self.name = 'Discriminator'

        self.initializer = tf.glorot_uniform_initializer(self.seed)

        self.kernel_size = 4

        self.kernels = [
            #(64, 2, 0),     # [batch, 32, 32, ch] => [batch, 16, 16, 64]
            (128, 2, 0),    # [batch, 16, 16, 64] => [batch, 8, 8, 128]
            (256, 2, 0),    # [batch, 8, 8, 128] => [batch, 4, 4, 256]
            (512, 1, 0),    # [batch, 4, 4, 256] => [batch, 4, 4, 512]
        ]

        self.variables = []

    def forward(self, X, reuse_vars=None):

        with tf.variable_scope(self.name, reuse=reuse_vars):

            output = tf.layers.Conv2D(
                                name='conv_1',
                                filters=64,
                                strides=2,
                                kernel_size=self.kernel_size,
                                padding='same',
                                kernel_initializer=self.initializer)(X)

            output = tf.nn.leaky_relu(output, name='leaky_ReLu_1')

            for i, kernel in enumerate(self.kernels):

                output = tf.layers.Conv2D(
                                name='conv_'+str(i+2),
                                filters=kernel[0],
                                strides=kernel[1],
                                kernel_size=self.kernel_size,
                                padding='same',
                                kernel_initializer=self.initializer)(output)

                output = tf.nn.leaky_relu(output, name='leaky_ReLu'+str(i+2))

            """output = tf.layers.Conv2D(
                                name='conv_' + str(i+3),
                                filters=1,
                                strides=1,
                                kernel_size=self.kernel_size,
                                padding='valid',
                                activation=None,
                                kernel_initializer=self.initializer)(output)"""
                
            output = tf.layers.Flatten(name='flatten')(output)
            output = tf.layers.Dense(
                                name='dense',
                                units=1,
                                activation=None,
                                kernel_initializer=self.initializer)(output)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return output

            