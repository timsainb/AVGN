import tensorflow as tf


class fc_net():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        self.encoder = [
            tf.keras.layers.InputLayer(input_shape=self.dims),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            #tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=self.n_Z),
        ]

        self.decoder = [
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            #tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=tf.math.reduce_prod(self.dims), activation="sigmoid"),
            tf.keras.layers.Reshape(target_shape=self.dims),
        ]

        self.discriminator = [
            #tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=1, activation=None),
        ]


class conv_net():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        self.encoder = [
            tf.keras.layers.InputLayer(input_shape=self.dims),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.n_Z * 2),
        ]

        self.decoder = [
            tf.keras.layers.Dense(units=7 * 7 * 32, activation="relu"),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation = "sigmoid"
            ),
        ]

        self.discriminator = [
            tf.keras.layers.InputLayer(input_shape=self.dims),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1, activation = None),
        ]


