import tensorflow as tf

class Discriminator:

    def __init__(self, seed):
        """
            Architecture:
                [?, 32, 32, ch] => [?, 16, 16, 64]
                [?, 16, 16, 64] => [?, 8, 8, 128]
                [?, 8, 8, 128] => [?, 4, 4, 256]
                [?, 4, 4, 256] => [?, 4, 4, 512]
                [?, 4, 4, 512] => [?, 1, 1, 1]

        """
        self.name = 'Discriminator'
        self.seed = seed

        self.initializer = tf.glorot_uniform_initializer(self.seed)

        self.is_training = True

        self.kernel_size = 4

        # (num_filters, strides)
        self.kernels = [
            (128, 2),
            (256, 2),
            (512, 1),
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

            output = tf.layers.Conv2D(
                                name='conv_' + str(i+3),
                                filters=1,
                                strides=1,
                                kernel_size=self.kernel_size,
                                padding='same',
                                activation=None,
                                kernel_initializer=self.initializer)(output)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return output

            