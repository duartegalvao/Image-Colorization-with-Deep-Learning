import tensorflow as tf

class Generator:

    def __init__(self, seed, is_training=True):
        """
            Architecture:
                Encoder: 
                    [?, 32, 32, input_ch] => [?, 32, 32, 64]
                    [?, 32, 32, 64] => [?, 16, 16, 128]
                    [?, 16, 16, 128] => [?, 8, 8, 256]
                    [?, 8, 8, 256] => [?, 4, 4, 512]
                    [?, 4, 4, 512] => [?, 2, 2, 512]

                Decoder:
                    [?, 2, 2, 512] => [?, 4, 4, 512]
                    [?, 4, 4, 512] => [?, 8, 8, 256]
                    [?, 8, 8, 256] => [?, 16, 16, 128]
                    [?, 16, 16, 128] => [?, 32, 32, 64]
                    [?, 32, 32, 64] => [?, 32, 32, out_ch]

        """
        self.name = 'Generator'
        self.seed = seed

        self.initializer = tf.glorot_uniform_initializer(self.seed)

        self.is_training = is_training

        self.kernel_size = 4

        # (num_filters, strides, dropout)
        self.kernels_encoder = [   
            (128, 2, 0),
            (256, 2, 0),
            (512, 2, 0.5),
            (512, 2, 0.5),
        ]

        # (num_filters, strides, dropout)
        self.kernels_decoder = [
            (512, 2, 0.5),
            (256, 2, 0.5),
            (128, 2, 0),
            (64, 2, 0),
        ]

        self.variables = []

    def forward(self, X, reuse_vars=None):

        with tf.variable_scope(self.name, reuse=reuse_vars):

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

                if kernel[2] != 0:
                    output = tf.keras.layers.Dropout(
                                    name='enc_dropout_' + str(i),
                                    rate=kernel[2],
                                    seed=self.seed)(output, training=self.is_training)


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

                if kernel[2] != 0:
                    output = tf.keras.layers.Dropout(
                                    name='dec_dropout_' + str(j),
                                    rate=kernel[2],
                                    seed=self.seed)(output, training=self.is_training)

                output = tf.concat([layers[len(layers)-j-2], output], axis=3)

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

            