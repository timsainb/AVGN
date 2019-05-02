import tensorflow as tf
import numpy as np
from tensorflow import layers


class ConvAE(object):
    def __init__(self, dims, batch_size, encoder_dims, decoder_dims, hidden_size, gpus=[], activation_fn=tf.nn.relu,
                 n_squishy=0, latent_loss='None', adam_eps=1.0, network_type='AE'):

        self.dims = dims
        self.batch_size = batch_size
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.hidden_size = hidden_size
        self.latent_loss = latent_loss  # either 'None', 'VAE', or 'distance'
        self.network_type = network_type
        self.batch_num = 0  # how many batches have been trained
        # training loss for SSE and distance
        self.loss_list = {
            'train': {
                'reconstruction': [],
                'latent': [],
            },
            'validation': {
                'reconstruction': [],
                'latent': [],
            },
        }

        self.default_activation = activation_fn
        self.adam_eps = adam_eps
        self.n_squishy = n_squishy
        self.num_gpus = len(gpus)  # number of GPUs to use
        if len(gpus) < 1:
            self.num_gpus = 1

        self.initialize_network()

    def initialization_AE(self):
        """ Initializes the network architecture of an autoencoder by
        1) running inputs through the network architecture
        2) calculating the losses for the architecture
        3) applying the losses to different parts of the network
        4) Creating a list of gradients for each GPU (or lack thereof)
        """

        # define learning rate and optimizers
        self.lr_D = tf.placeholder(tf.float32, shape=[])
        self.lr_E = tf.placeholder(tf.float32, shape=[])
        self.opt_D = tf.train.AdamOptimizer(learning_rate=self.lr_D, epsilon=self.adam_eps)
        self.opt_E = tf.train.AdamOptimizer(learning_rate=self.lr_E, epsilon=self.adam_eps)

        # placeholder for latent loss weight (importance of latent space constraint)
        # Placeholder for weight of distance metric
        self.latent_loss_weights = tf.placeholder(tf.float32)


        # Construct the model
        # self.inference_AE(self.x_input)
        # encoder
        with tf.variable_scope("enc"):
            # (z_log_sigma_sq is just for vae)
            self.enc_net, enc_shapes, self.z_log_sigma_sq = self.encoder(self.x_input)  # get z from the input

            if self.latent_loss != 'VAE':
                self.z_x = self.enc_net[-1]

            else:
                self.z_x_mean = self.enc_net[-1]
                eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1, dtype=tf.float32)
                self.z_x = self.z_x_mean  # + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # decoder
        with tf.variable_scope("dec"):
            self.dec_net = self.decoder(self.z_x, enc_shapes)  # get output from z
            self.x_tilde = self.dec_net[-1]

        # Calculate the loss for this tower
        if self.latent_loss == 'distance':
            self.distance_loss = distance_loss(self.x_input, self.z_x)
            self.recon_loss = tf.reduce_mean(tf.square(self.x_input - self.x_tilde))
            self.L_e = tf.clip_by_value(
                self.recon_loss + self.latent_loss_weights*self.distance_loss, -1e10, 1e10)
        elif self.latent_loss == 'VAE':
            self.recon_loss = -tf.reduce_sum(self.x_input * tf.log(1e-8 + self.x_tilde) +
                                             (1-self.x_input) * tf.log(1e-8 + 1 - self.x_tilde), 1)
            self.KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                              - tf.square(self.z_x_mean)
                                                              - tf.exp(self.z_log_sigma_sq), 1))
            self.L_e = tf.clip_by_value(
                self.recon_loss + self.latent_loss_weights*self.KL_loss, -1e10, 1e10)
        else:  # is regular autoencoder
            self.recon_loss = tf.reduce_mean(tf.square(self.x_input - self.x_tilde))
            self.L_e = tf.clip_by_value(self.recon_loss, -1e10, 1e10)
        self.L_d = tf.clip_by_value(self.recon_loss, -1e10, 1e10)

        # specify loss to parameters
        self.params = tf.trainable_variables()
        self.E_params = [i for i in self.params if 'enc/' in i.name]
        self.D_params = [i for i in self.params if 'dec/' in i.name]


        # Calculate the gradients for the batch of data on this CIFAR tower.
        self.grads_e = self.opt_E.compute_gradients(self.L_e, var_list=self.E_params)
        self.grads_d = self.opt_D.compute_gradients(self.L_d, var_list=self.D_params)

        self.train_E = self.opt_E.apply_gradients(self.grads_e, global_step=self.global_step)
        self.train_D = self.opt_D.apply_gradients(self.grads_d, global_step=self.global_step)

        if self.latent_loss == 'VAE': self.recon_loss = tf.reduce_sum(self.recon_loss)

    def initialize_network(self,):
        """ Defines the network architecture
        """
        # initialize graph and session
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph=self.graph, config=self.config)

        # Global step needs to be defined to coordinate multi-GPU
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Placeholder for input data
        self.x_input = tf.placeholder(
            tf.float32, [self.batch_size*self.num_gpus, np.prod(self.dims)])

        if self.network_type == 'AE':
            self.initialization_AE()
        elif self.network_type == 'GAIA':
            self.initialization_GAIA()

        # Start the Session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # initialize network saver
        print('Network Initialized')

    def encoder(self, X, verbose=True):
        """ Draws the encoder of the network
        """
        enc_net = [tf.reshape(X, [self.batch_size, self.dims[0], self.dims[1], self.dims[2]])]
        for lay_num, (filters, kernel_size, stride) in enumerate(self.encoder_dims):
            if kernel_size > 0:  # if this is a convolutional layer
                if lay_num == len(self.encoder_dims)-1:  # if this is the last layer

                    enc_net.append(tf.contrib.layers.flatten(layers.conv2d(enc_net[len(enc_net)-1], filters=filters,
                                                                           kernel_size=kernel_size, strides=stride, padding='same',
                                                                           name='enc_'+str(lay_num),
                                                                           activation=self.default_activation)))
                else:
                    if self.encoder_dims[lay_num + 1][1] == 0:
                        # flatten this layer
                        enc_net.append(tf.contrib.layers.flatten(layers.conv2d(enc_net[len(enc_net)-1], filters=filters,
                                                                               kernel_size=kernel_size, strides=stride, padding='same',
                                                                               name='enc_' +
                                                                               str(lay_num),
                                                                               activation=self.default_activation)))
                    else:
                        enc_net.append(layers.conv2d(enc_net[len(enc_net)-1], filters=filters, kernel_size=kernel_size,
                                                     strides=stride, padding='same', name='enc_'+str(lay_num),
                                                     activation=self.default_activation))
            else:
                enc_net.append(layers.dense(enc_net[len(enc_net)-1], units=filters, name='enc_'+str(lay_num),
                                            activation=self.default_activation))
        enc_shapes = [shape(i) for i in enc_net]
        # append latent layer
        if self.latent_loss == 'VAE':
            z_log_sigma_sq = layers.dense(
                enc_net[len(enc_net)-1], units=self.hidden_size, activation=None, name='z_log_sigma_squared')
        else:
            z_log_sigma_sq = 0
        enc_net.append(layers.dense(
            enc_net[len(enc_net)-1], units=self.hidden_size, activation=None, name='latent_layer'))  # 32, 2
        if verbose:
            print('Encoder shapes: ', enc_shapes)
        return enc_net, enc_shapes, z_log_sigma_sq

    def decoder(self, Z, verbose=True):
        """ Draws the decoder fo the network
        """
        dec_net = [Z]
        prev_dec_shape = None
        num_div = len([stride for lay_num, (filters, kernel_size, stride)
                       in enumerate(self.decoder_dims) if stride == 2])
        cur_shape = int(self.dims[1]/(2**(num_div-1)))

        for lay_num, (filters, kernel_size, stride) in enumerate(self.decoder_dims):
            #print( [i for i in tf.trainable_variables() if 'generator/' in i.name])
            if kernel_size > 0:  # if this is a convolutional layer

                # this is the first layer and the first convolutional layer
                if (lay_num == 0) or (self.decoder_dims[lay_num - 1][1] == 0):
                    dec_net.append(tf.reshape(layers.dense(dec_net[len(dec_net)-1], cur_shape*cur_shape*filters, name='dec_'+str(lay_num),
                                                           activation=self.default_activation),
                                              [self.batch_size,  cur_shape, cur_shape, filters]))
                elif stride == 2:  # if the spatial size of the previous layer is greater than the image size of the current layer
                    # we need to resize the current network dims
                    cur_shape *= 2
                    dec_net.append(tf.image.resize_nearest_neighbor(
                        dec_net[len(dec_net)-1], (cur_shape, cur_shape)))

                elif lay_num == len(self.decoder_dims)-1:  # if this is the last layer
                    # append a normal layer

                    dec_net.append((layers.conv2d(dec_net[len(dec_net)-1], filters=filters, kernel_size=kernel_size,
                                                  strides=1, padding='same', name='dec_'+str(lay_num),
                                                  activation=self.default_activation)))

                # If the next layer is not convolutional but this one is
                elif self.decoder_dims[lay_num + 1][1] == 0:
                    # flatten this layers
                    dec_net.append(tf.contrib.layers.flatten(layers.conv2d(dec_net[len(dec_net)-1], filters=filters,
                                                                           kernel_size=kernel_size, strides=1, padding='same',
                                                                           name='dec_'+str(lay_num),
                                                                           activation=self.default_activation)))
                else:
                    # append a normal layer
                    dec_net.append((layers.conv2d(dec_net[len(dec_net)-1], filters=filters, kernel_size=kernel_size,
                                                  strides=1, padding='same', name='dec_'+str(lay_num),
                                                  activation=self.default_activation)))
            else:  # if this is a dense layer
                # append the dense layer
                dec_net.append(layers.dense(dec_net[len(dec_net)-1], units=filters, name='dec_'+str(lay_num),
                                            activation=self.default_activation))

                # append the output layer

        if (self.dims[0] != shape(dec_net[-1])[1]) & (self.dims[1] != shape(dec_net[-1])[2]):
            print('warning: shape does not match image shape')
            dec_net.append(tf.image.resize_nearest_neighbor(
                dec_net[len(dec_net)-1], (self.dims[0], self.dims[1])))
        dec_net.append(layers.conv2d(
            dec_net[len(dec_net)-1], self.dims[2], 1, strides=1, activation=tf.sigmoid, name='output_layer'))
        dec_net.append(tf.contrib.layers.flatten(dec_net[len(dec_net)-1]))

        if verbose:
            print('Decoder shapes: ', [shape(i) for i in dec_net])
        return dec_net

    def _get_tensor_by_name(self, tensor_list):
        return [self.graph.get_tensor_by_name(i) for i in tensor_list]

    def save_network(self, save_location, verbose=True):
        """ Save the network to some location"""
        self.saver.save(self.sess, ''.join([save_location]))
        if verbose:
            print('Network Saved')

    def load_network(self, load_location, verbose=True):
        """ Retrieve the network from some location"""
        self.saver = tf.train.import_meta_graph(load_location + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(
            '/'.join(load_location.split('/')[:-1]) + '/'))
        if verbose:
            print('Network Loaded')


def shape(tensor):
    """ get the shape of a tensor
    """
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def squared_dist(A):
    """
    Computes the pairwise distance between points
    #http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """

    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
    return distances



def distance_loss(x, z_x):
    """ Loss based on the distance between elements in a batch
    """
    sdx = squared_dist(x)
    sdx = sdx/tf.reduce_mean(sdx)
    sdz = squared_dist(z_x)
    sdz = sdz/tf.reduce_mean(sdz)
    return tf.reduce_mean(tf.square(tf.log(tf.constant(1.)+sdx) - (tf.log(tf.constant(1.)+sdz))))
