import tensorflow as tf
from tqdm.autonotebook import tqdm


class WGAN(tf.keras.Model):
    """[summary]
    
    references
    # https://github.com/ilguyi/gans.tensorflow.v2
    # https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/
    
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.__dict__.update(kwargs)

        self.gen = tf.keras.Sequential(self.gen)
        self.disc = tf.keras.Sequential(self.disc)

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def sample(self, eps=None, n_samp=100):
        if eps is None:
            eps = tf.random.normal(shape=(n_samp, self.n_Z))
        x_samp = self.generate(eps)
        return x_samp

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        # generating noise from a uniform distribution

        z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])

        # run noise through generator
        x_gen = self.generate(z_samp)
        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)
        
        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        ### losses
        disc_loss = (tf.reduce_mean(logits_x) - tf.reduce_mean(logits_x_gen) + d_regularizer * self.gradient_penalty_weight)

        # losses of fake with label "1"
        gen_loss = tf.reduce_mean(logits_x_gen)

        

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def train(self, train_dataset):
        for train_x in tqdm(train_dataset, leave=False):
            gen_gradients, disc_gradients = self.compute_gradients(train_x)
            self.apply_gradients(gen_gradients, disc_gradients)




def test_network(model, test_dataset):
    D_loss = tf.keras.metrics.Mean()
    G_loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        disc_loss, gen_loss = model.compute_loss(test_x)
        D_loss(disc_loss)
        G_loss(gen_loss)

    return D_loss.result(), G_loss.result()

