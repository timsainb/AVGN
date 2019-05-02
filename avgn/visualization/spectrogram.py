import matplotlib.pyplot as plt
import numpy as np


def plot_spec(spec, fig, ax, extent=None, cmap=plt.cm.afmhot, show_cbar=True):
    """plot spectrogram
    
    [description]
    
    Arguments:
        spec {[type]} -- [description]
        fig {[type]} -- [description]
        ax {[type]} -- [description]
    
    Keyword Arguments:
        cmap {[type]} -- [description] (default: {plt.cm.afmhot})
    """
    spec_ax = ax.matshow(
        spec,
        interpolation=None,
        aspect="auto",
        cmap=plt.cm.afmhot,
        origin="lower",
        extent=extent,
    )
    if show_cbar:
        cbar = fig.colorbar(spec_ax, ax=ax)
        return spec_ax, cbar
    else:
        return spec_ax


def visualize_spec(wav_spectrogram, save_loc=None, show=True, figsize=(20, 5)):
    """basic spectrogram visualization and saving
    
    [description]
    
    Arguments:
        wav_spectrogram {[type]} -- [description]
    
    Keyword Arguments:
        save_loc {[type]} -- [description] (default: {None})
        show {bool} -- [description] (default: {True})
        figsize {tuple} -- [description] (default: {(20,5)})
    """
    fig, ax = plt.subplots(figsize=figsize)
    spec_ax, cbar = plot_spec(wav_spectrogram, fig, ax)
    if show:
        plt.show()
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches="tight")
        plt.close()


def plot_segmentations(
    spec,
    vocal_envelope,
    all_syllable_starts,
    all_syllable_lens,
    fft_rate,
    hparams,
    fig = None,
    axs = None,
    figsize=(60, 9),
):
    """Plot the segmentation points over a spectrogram
    
    [description]
    
    Arguments:
        spec {[type]} -- [description]
        vocal_envelope {[type]} -- [description]
        all_syllable_start {[type]} -- [description]
        all_syllable_lens {[type]} -- [description]
        fft_rate {[type]} -- [description]
    
    Keyword Arguments:
        figsize {tuple} -- [description] (default: {(60, 9)})
    """
    stop_time = np.shape(spec)[1] / fft_rate
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=figsize)
    extent = [0, stop_time, 0, hparams["sample_rate"] / 2]
    plot_spec(spec, fig, axs[0], extent=extent, show_cbar=False)

    # plot segmentation marks
    for st, slen in zip(all_syllable_starts, all_syllable_lens):
        axs[0].axvline(st, color="w", linestyle="-", lw=3, alpha=0.75)
        axs[0].axvline(st + slen, color="w", linestyle="-", lw=3, alpha=0.75)

    axs[1].plot(vocal_envelope, color="k", lw=4)
    axs[1].set_xlim([0, len(vocal_envelope)])

    plt.show()


def plot_syllable_list(
    all_syllables,
    n_mel_freq_components,
    max_rows=3,
    max_sylls=100,
    width=400,
    zoom=1,
    spacing=1,
    cmap=plt.cm.viridis,
):
    """Plot a list of syllables as one large canvas
    
    [description]
    
    Arguments:
        all_syllables {[type]} -- [description]
        hparams {[type]} -- [description]
    
    Keyword Arguments:
        max_rows {number} -- [description] (default: {3})
        max_sylls {number} -- [description] (default: {100})
        width {number} -- [description] (default: {400})
        zoom {number} -- [description] (default: {1})
        spacing {number} -- [description] (default: {1})
        cmap {[type]} -- [description] (default: {plt.cm.viridis})
    """
    canvas = np.zeros((n_mel_freq_components * max_rows, width))
    x_loc = 0
    row = 0

    for i, syll in enumerate(all_syllables):

        # if the syllable is too long
        if np.shape(syll)[1] > width:
            continue

        if (x_loc + np.shape(syll)[1]) > width:
            if row == max_rows - 1:
                break

            else:
                row += 1
                x_loc = 0

        canvas[
            row * n_mel_freq_components : (row + 1) * n_mel_freq_components,
            x_loc : (x_loc + np.shape(syll)[1]),
        ] = np.flipud(syll)

        x_loc += np.shape(syll)[1] + spacing

    if row < max_rows:
        canvas = canvas[: (row + 1) * n_mel_freq_components]

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(width / 32 * zoom, max_rows * zoom)
    )
    ax.matshow(
        canvas,
        cmap=cmap,
        # origin='lower',
        aspect="auto",
        interpolation="nearest",
    )
    plt.show()

def plot_bout_to_syllable_pipeline(
    data,
    vocal_envelope,
    wav_spectrogram,
    all_syllables,
    all_syllable_starts, 
    all_syllable_lens,
    rate,
    fft_rate,
    zoom=1,
    submode=True,
    figsize=(50, 10),
    ):
    """plots the whole plot_bout_to_syllable_pipeline pipeline
    """
    # create a plot where the top is waveform, underneath is spectrogram, underneath is segmented syllables
    # fig = plt.subplots(figsize=figsize)
    # gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,2,1])
    # ax=[plt.subplot(i) for i in gs]
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    # plot the original vocalization data
    ax[0].plot(data, color="black")
    ax[0].set_xlim([0, len(data)])
    ax[0].axis("off")
    ax[0].set_ylim([np.min(data), np.max(data)])

    # plot the vocal envelope below the data
    ax[1].plot(vocal_envelope)
    ax[1].set_xlim([0, len(vocal_envelope)])
    ax[1].set_ylim([np.min(vocal_envelope), np.max(vocal_envelope)])
    ax[1].axis("off")

    stop_time = np.shape(wav_spectrogram)[1] / fft_rate
    extent = [0, stop_time, 0, rate / 2]
    plot_spec(wav_spectrogram, fig, ax[2], extent=extent, show_cbar=False)

    # plot segmentation marks
    for st, slen in zip(all_syllable_starts, all_syllable_lens):
        ax[2].axvline(st, color="w", linestyle="-", lw=3, alpha=0.75)
        ax[2].axvline(st + slen, color="w", linestyle="-", lw=3, alpha=0.75)


    """    for si, syll_se in enumerate([(i[0], i[-1]) for i in all_syllables_time_idx]):
        imscatter(
            (syll_se[1] + syll_se[0]) / 2,
            0,
            np.flipud(norm(all_syllables[si])),
            zoom=zoom,
            ax=ax[3],
        )
        ax[3].text(
            (syll_se[1] + syll_se[0]) / 2,
            0.15,
            round(syllable_lengths[si], 3),
            fontsize=15,
            horizontalalignment="center",
        )"""

    ax[3].set_xlim([0, len(data)])
    ax[3].set_ylim([-0.2, 0.2])
    ax[3].axis("off")
    plt.tight_layout()
    plt.show()

