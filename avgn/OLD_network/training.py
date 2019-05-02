import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import seaborn as sns
import h5py

# import local methods from the source code
from avgn.network_analysis.network_analysis import *


def load_from_hdf5(hdf_locs, to_load, min_ex=1, verbose=False):
    """Loads content from a list of HDF5 files"""
    hdf5_content = {}
    with h5py.File(hdf_locs[0], 'r') as hf:
        for tl in to_load:
            hdf5_content[tl] = hf[tl].value
        hdf5_content['name'] = np.repeat(list(hf.attrs.values())[0],
                                         np.shape(hf['spectrograms'].value)[0])

    for i, folder in enumerate(hdf_locs[1:]):
        with h5py.File(folder, 'r') as hf:
            if len(hf[to_load[0]].value) < min_ex:
                continue
            if verbose: print(folder, len(hf[to_load[0]].value))
            for tl in to_load:
                hdf5_content[tl] = np.append(hdf5_content[tl], hf[tl].value, axis=0)
            hdf5_content['name'] = np.append(hdf5_content['name'], np.repeat(
                list(hf.attrs.values())[0], np.shape(hf['spectrograms'].value)[0]))
    return hdf5_content

# Function for creating an iterator


def data_iterator(x, y=None, batch_size=False, num_gpus=1, dims=[10, 10], randomize=True):
    """ A simple data iterator used to load the spectrograms
     """
    if batch_size == False:
        batch_size = len(x)
    while True:
        idxs = np.arange(0, len(x))
        if randomize:
            np.random.shuffle(idxs)
        for batch_idx in np.arange(0, len(x)-batch_size*num_gpus, batch_size*num_gpus):
            cur_idxs = idxs[batch_idx:batch_idx+batch_size*num_gpus]
            batch = x[cur_idxs].reshape((batch_size*num_gpus, np.prod(dims))).astype('float32')
            batch_y = y[cur_idxs] if y else None  # if there is a Y, grab it
            yield batch/255., batch_y


# Function for training the networks
def train_AE(model, iter_, dataset_size=1, latent_loss_weights=1.0e-1,
             validation_iter_=False, validation_size=0, learning_rate=1.0, return_list=[]):
    """
    Training an autoencoder network. NOTE: this will not work with GAN network architectures. Write a different function for that.
    """
    # how many batches to iterate over
    total_batch = int(np.floor(dataset_size/model.batch_size/model.num_gpus))

    tl = [getattr(model, i) for i in return_list]

    # train batches
    training_rows = []  # saves information about training
    for batch in tqdm(np.arange(total_batch), leave=False):
        model.batch_num += 1
        next_batches = iter_.__next__()[0]
        # run the epoch
        training_rows.append([model.batch_num] +
                             model.sess.run(tl,
                                            {
                                                model.x_input: next_batches,
                                                model.latent_loss_weights: latent_loss_weights,
                                                model.lr_D: learning_rate,
                                                model.lr_E: learning_rate,
                                            }
                                            ))

    # how many validation batches to iterate over
    if validation_iter_ != False:
        total_val_batch = int(np.floor((validation_size)/model.batch_size/model.num_gpus))
        val_rows = []  # saves information about training
        for batch in tqdm(np.arange(total_val_batch), leave=False):
            next_batches = validation_iter_.__next__()[0]
            # run the epoch
            val_rows.append([model.batch_num] +
                            model.sess.run(tl[2:],
                                           {
                                model.x_input: next_batches,
                                model.latent_loss_weights: latent_loss_weights,
                                model.lr_D: learning_rate,
                                model.lr_E: learning_rate,
                            }
            ))

        return pd.DataFrame(training_rows, columns=['batch'] + return_list), pd.DataFrame(val_rows[2:], columns=['batch'] + return_list[2:])
    else:
        return pd.DataFrame(training_rows, columns=['batch'] + return_list), None


# function for visualizing the network in training
def visualize_2D_AE(model, training_df, validation_df, example_data, num_examples, batch_size, num_gpus, dims, iter_, n_cols=4, std_to_plot=2.5, summary_density=50, save_loc=False, n_samps_per_dim=8):
    """
    Visualization of AE as it trains in 2D space
    """
    # choose a color palette
    current_palette = sns.color_palette()
    # summarize training
    bins = [0] + np.unique(np.logspace(0, np.log2(np.max(training_df.batch +
                                                         [100])), num=summary_density, base=2).astype('int'))
    training_df['batch_bin'] = pd.cut(training_df.batch+1, bins, labels=bins[:-1])
    training_summary = training_df.groupby(['batch_bin']).describe()
    validation_df['batch_bin'] = pd.cut(validation_df.batch+1, bins, labels=bins[:-1])
    validation_summary = validation_df.groupby(['batch_bin']).describe()
    validation_df[:3]

    # get reconstructions of example data
    example_recon, z = model.sess.run((model.x_tilde, model.z_x), {model.x_input: example_data})
    # get Z representations of data
    z = np.array(generate_manifold(model, dims, iter_, num_examples, batch_size, num_gpus))

    if np.shape(z)[1] == 2:
        # generate volumetric data
        x_bounds = [-inv_z_score(std_to_plot, z[:, 0]), inv_z_score(std_to_plot, z[:, 0])]
        y_bounds = [-inv_z_score(std_to_plot, z[:, 1]), inv_z_score(std_to_plot, z[:, 1])]
        maxx, maxy, hx, hy, pts = make_grid(x_bounds, y_bounds, maxx=int(
            n_samps_per_dim), maxy=int(n_samps_per_dim))

        dets = metric_and_volume(model, maxx, maxy, hx, hy, pts, dims, batch_size)

    fig = plt.figure(figsize=(10, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    scatter_ax = plt.Subplot(fig, outer[0])
    scatter_ax.scatter(z_score(z[:, 0]), z_score(z[:, 1]), alpha=.1, s=3, color='k')
    scatter_ax.axis('off')
    scatter_ax.set_xlim([-std_to_plot, std_to_plot])
    scatter_ax.set_ylim([-std_to_plot, std_to_plot])
    fig.add_subplot(scatter_ax)

    if np.shape(z)[1] == 2:
        volume_ax = plt.Subplot(fig, outer[1])
        volume_ax.axis('off')
        volume_ax.matshow(np.log2(dets), cmap=plt.cm.viridis)
        fig.add_subplot(volume_ax)

    recon_ax = gridspec.GridSpecFromSubplotSpec(int(n_cols), int(n_cols/2),
                                                subplot_spec=outer[2], wspace=0.1, hspace=0.1)

    for axi in range(int(n_cols) * int(n_cols/2)):
        recon_sub_ax = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                        subplot_spec=recon_ax[axi], wspace=0.1, hspace=0.1)
        orig_ax = plt.Subplot(fig, recon_sub_ax[0])
        orig_ax.matshow(np.squeeze(example_data[axi].reshape(dims)), origin='lower')
        orig_ax.axis('off')
        rec_ax = plt.Subplot(fig, recon_sub_ax[1])
        rec_ax.matshow(np.squeeze(example_recon[axi].reshape(dims)), origin='lower')
        rec_ax.axis('off')
        fig.add_subplot(orig_ax)
        fig.add_subplot(rec_ax)

    error_ax = plt.Subplot(fig, outer[3])
    #error_ax.plot(training_df.batch, training_df.recon_loss)
    training_plt, = error_ax.plot(training_summary.recon_loss['mean'].index.astype('int').values,
                                  training_summary.recon_loss['mean'].values, alpha=1, color=current_palette[0], label='training')

    error_ax.fill_between(training_summary.recon_loss['mean'].index.astype('int').values,
                          training_summary.recon_loss['mean'].values -
                          training_summary.recon_loss['std'].values,
                          training_summary.recon_loss['mean'].values + training_summary.recon_loss['std'].values, alpha=.25, color=current_palette[0])

    error_ax.fill_between(validation_summary.recon_loss['mean'].index.astype('int').values,
                          validation_summary.recon_loss['mean'].values -
                          validation_summary.recon_loss['std'].values,
                          validation_summary.recon_loss['mean'].values + validation_summary.recon_loss['std'].values, alpha=.25, color=current_palette[1])
    validation_plt, = error_ax.plot(validation_summary.recon_loss['mean'].index.astype('int').values,
                                    validation_summary.recon_loss['mean'].values - validation_summary.recon_loss['std'].values, alpha=1,
                                    color=current_palette[1], label='validation')

    error_ax.legend(handles=[validation_plt, training_plt], loc=1)
    error_ax.set_yscale("log")
    error_ax.set_xscale("log")
    fig.add_subplot(error_ax)
    if save_loc != False:
        if not os.path.exists('/'.join(save_loc.split('/')[:-1])):
            os.makedirs('/'.join(save_loc.split('/')[:-1]))
        plt.savefig(save_loc)
    plt.show()
