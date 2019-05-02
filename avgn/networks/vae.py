import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm

class VAE(tf.keras.Model):
    """a basic vae class for tensorflow
    
    references
    - https://www.tensorflow.org/alpha/tutorials/generative/cvae
    
    Extends:
        tf.keras.Model
    """
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)
        
        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)
    
    def encode(self, x):
        mean, logvar = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.dec(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def sample(self, eps=None, n_samp=100):
        if eps is None:
            eps = tf.random.normal(shape=(n_samp, self.n_Z))
        return self.decode(eps, apply_sigmoid=True)
    
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)
    
    #@tf.function
    def train(self, train_dataset):
        for train_x in tqdm(train_dataset, leave=False): 
            gradients = self.compute_gradients(train_x)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) 


#@tf.function
def test_network(model, test_dataset):
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(model.compute_loss(test_x))
    elbo = -loss.result()
    return elbo


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )
