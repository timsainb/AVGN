#import numpy as np
import tensorflow as tf
#from tqdm.autonotebook import tqdm
from tensorflow.keras import Model
from tensorflow_probability.python.distributions import Chi2

class GAIA(tf.keras.Model):
    """a basic vae class for tensorflow
    
    references
    - https://www.tensorflow.org/alpha/tutorials/generative/cvae
    
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(GAIA, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

        inputs, outputs = self.unet_function()
        self.disc = Model(inputs=[inputs], outputs=[outputs])
        self.chsq = Chi2(df=1/self.batch_size)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def discriminate(self, x):
        return self.disc(x)

    def regularization(self, x1, x2):
        return tf.reduce_mean(tf.square(x1 - x2))

    @tf.function
    def network_pass(self, x):
        z = self.encode(x)
        xg = self.decode(z)
        zi = self._interpolate_z(z)
        xi = self.decode(zi)
        d_xi = self.discriminate(xi)
        d_x = self.discriminate(x)
        d_xg = self.discriminate(xg)
        return z, xg, zi, xi, d_xi, d_x, d_xg

    @tf.function
    def compute_loss(self, x):
        # run through network
        z, xg, zi, xi, d_xi, d_x, d_xg = self.network_pass(x)

        # compute losses
        xg_loss = self.regularization(x, xg)
        d_xg_loss = self.regularization(x, d_xg)
        d_xi_loss = self.regularization(xi, d_xi)
        d_x_loss = self.regularization(x, d_x)
        D_prop = sigmoid(d_x_loss - d_xi_loss, mult=20)

        return D_prop, d_xg_loss, d_xi_loss, d_x_loss, xg_loss

    
    def compute_gradients(self, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            D_prop, d_xg_loss, d_xi_loss, d_x_loss, xg_loss = self.compute_loss(x)

            gen_loss =  d_xg_loss + xg_loss + d_xi_loss*self.alpha
            disc_loss = d_xg_loss + d_x_loss  - d_xi_loss*self.alpha

        # balance learning rates
        #self.gen_optimizer.lr = (tf.constant(1.0) - D_prop)*self.lr
        #self.disc_optimizer.lr = D_prop*self.lr


        gen_gradients = gen_tape.gradient(
            gen_loss, self.enc.trainable_variables + self.dec.trainable_variables
        )
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        return gen_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, gen_gradients, disc_gradients):
        self.gen_optimizer.apply_gradients(
            zip(
                gen_gradients,
                self.enc.trainable_variables + self.dec.trainable_variables,
            )
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def train(self, x):
        gen_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(gen_gradients, disc_gradients)

    def _interpolate_z(self, z):
        """ takes the dot product of some random tensor of batch_size,
         and the z representation of the batch as the interpolation
        """
        if self.chsq.df != z.shape[0]:
            self.chsq = Chi2(df=1/z.shape[0])
        ip = self.chsq.sample((z.shape[0], z.shape[0]))
        ip = ip / tf.reduce_sum(ip, axis=0)
        zi = tf.transpose(tf.tensordot(tf.transpose(z), ip, axes=1))
        return zi


def sigmoid(x, shift=0.0, mult=20):
    """ squashes a value with a sigmoid
    """
    return tf.constant(1.0) / (
        tf.constant(1.0) + tf.exp(-tf.constant(1.0) * (x * mult))
    )
