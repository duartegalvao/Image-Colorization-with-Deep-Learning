import tensorflow as tf

class Generator:

    def __init__(self, seed):
        self.seed = seed
        self.name = 'Generator'

        self.initializer = tf.glorot_uniform_initializer(self.seed)

        self.kernel_size = 4

        self.kernels_encoder = [
            #(64, 1, 0),     # [batch, 32, 32, ch] => [batch, 32, 32, 64]
            (128, 2, 0),    # [batch, 32, 32, 64] => [batch, 16, 16, 128]
            (256, 2, 0),    # [batch, 16, 16, 128] => [batch, 8, 8, 256]
            (512, 2, 0),    # [batch, 8, 8, 256] => [batch, 4, 4, 512]
            (512, 2, 0),    # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        ]

        self.kernels_decoder = [
            (512, 2, 0.5),  # [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (256, 2, 0.5),  # [batch, 4, 4, 512] => [batch, 8, 8, 256]
            (128, 2, 0),    # [batch, 8, 8, 256] => [batch, 16, 16, 128]
            (64, 2, 0),     # [batch, 16, 16, 128] => [batch, 32, 32, 64]
        ]

        self.variables = []

    def forward(self, X):

        with tf.variable_scope(self.name, reuse=None):

            layers = []

            output = tf.layers.Conv2D(
                                name='enc_conv_1',
                                filters=64,
                                strides=1,
                                kernel_size=self.kernel_size,
                                padding='same',
                                kernel_initializer=self.initializer)(X)

            output = tf.layers.BatchNormalization(name='enc_bn_1')(output)

            output = tf.nn.leaky_relu(output, name='enc_leaky_ReLu_1')

            layers.append(output)

            for i, kernel in enumerate(self.kernels_encoder):

                output = tf.layers.Conv2D(
                                name='enc_conv_'+str(i+2),
                                filters=kernel[0],
                                strides=kernel[1],
                                kernel_size=self.kernel_size,
                                padding='same',
                                kernel_initializer=self.initializer)(output)

                output = tf.layers.BatchNormalization(name='enc_bn_'+str(i+2))(output)

                output = tf.nn.leaky_relu(output, name='enc_leaky_ReLu'+str(i+2))

                layers.append(output)

            for j, kernel in enumerate(self.kernels_decoder):

                output = tf.layers.Conv2DTranspose(
                                name='dec_conv_t_'+str(j+1),
                                filters=kernel[0],
                                strides=kernel[1],
                                kernel_size=self.kernel_size,
                                padding='same',
                                kernel_initializer=self.initializer)(output)

                output = tf.layers.BatchNormalization(name='dec_bn_' + str(i+3+j))(output)

                output = tf.nn.relu(output, name='dec_ReLu_'+str(j+1))

                if kernel[2] > 0:
                    output = tf.layers.Dropout(
                                    name='dec_dropout_' + str(j),
                                    rate=kernel[2],
                                    seed=self.seed)(output)

                output = tf.concat([layers[len(layers) - j - 2], output], axis=3)

            output = tf.layers.Conv2D(
                                name='dec_conv_' + str(i+3),
                                filters=2,
                                strides=1,
                                kernel_size=1,
                                padding='same',
                                activation=tf.nn.tanh,
                                kernel_initializer=self.initializer)(output)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            
        return output

            