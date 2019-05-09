#import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm

class AE(tf.keras.Model):
    """a basic autoencoder class for tensorflow
    
    references
    - https://www.tensorflow.org/alpha/tutorials/generative/cvae
    
    Extends:
        tf.keras.Model
    """
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.__dict__.update(kwargs)
        
        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    @tf.function
    def encode(self, x):
        return self.enc(x)

    @tf.function
    def decode(self, z):
        return self.dec(z)
    
    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        return ae_loss
    
    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

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
