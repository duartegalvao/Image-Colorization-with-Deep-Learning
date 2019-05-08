import tensorflow as tf

class UNet:

    def __init__(self, seed):
        self.seed = seed
        self.initializer = tf.glorot_uniform_initializer(self.seed)

        self.kernel_size = 4

        self.kernels_encoder = [
            (64, 1, 0),     # [batch, 32, 32, ch] => [batch, 32, 32, 64]
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

    def forward(self, X):

        output = X

        for i, kernel in enumerate(self.kernels_encoder):

            name = 'conv' + str(i)
            output = tf.keras.layers.Conv2D(
                            name=name,
                            filters= kernel[0],
                            padding='same',
                            kernel_size=self.kernel_size,
                            strides= kernel[1],
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer)(output)

        for j, kernel in enumerate(self.kernels_decoder):

            name = 'conv_tranpose' + str(j)
            output = tf.keras.layers.Conv2DTranspose(
                            name=name,
                            filters=kernel[0],
                            padding='same',
                            kernel_size=self.kernel_size,
                            strides=kernel[1],
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer)(output)

        name = 'conv_last'
        output = tf.keras.layers.Conv2D(
                            name=name,
                            filters=3,
                            kernel_size=1,
                            padding='same',
                            strides=1,
                            activation=tf.nn.tanh,
                            kernel_initializer=self.initializer)(output)
        
        return output

        