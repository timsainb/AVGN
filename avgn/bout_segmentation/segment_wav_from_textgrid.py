# create syllable hdf5 dataset from waveforms + textgrids

import numpy as np
from PIL import Image  # for resizing syllables
import pandas as pd
from datetime import datetime
from praatio import tgio

from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.bout_segmentation.dynamic_threshold_segmentation import dynamic_spectrogram
from avgn.utils.general import zero_one_norm
from avgn.visualization.spectrogram import (
    plot_syllable_list,
    plot_bout_to_syllable_pipeline,
)
from avgn.utils.audio import load_wav


def cut_syllables(onsets, offsets, spec, fft_rate, hparams):
    """Cut spectrogram into syllables based upon onsets and offsets
    
    [description]
    
    Arguments:
        onsets {[type]} -- [description]
        offsets {[type]} -- [description]
        spec {[type]} -- [description]
        fft_rate {[type]} -- [description]
        params {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Cut into syllables
    all_syllables = []
    all_syllable_starts = []
    all_syllable_lens = []

    # for each onset offset pair
    for on, off in zip(onsets, offsets):
        # if the syllable is sufficiently long
        if off > on + hparams["min_syll_len_s"]:
            all_syllables.append(spec[:, on : off + 1])
            all_syllable_starts.append(on / fft_rate)
            all_syllable_lens.append((off - on) / fft_rate)

    return all_syllables, all_syllable_starts, all_syllable_lens


def resize_syllables(all_syllables, fft_rate, hparams):
    """[summary]
    
    [description]
    
    Arguments:
        all_syllables {[type]} -- [description]
        hparams {[type]} -- [description]
    """
    for i, syll in enumerate(all_syllables):
        # resize the syllable
        resize_shape = [
            int((np.shape(syll)[1] / fft_rate) * hparams["resize_samp_fr"]),
            hparams["num_freq_final"],
        ]
        # if the width is greater than the expected length of the syllable
        if resize_shape[0] > hparams["max_size_syll"]:
            # set the width to the expected length
            resize_shape[0] = hparams["max_size_syll"]
        # perform resizing
        all_syllables[i] = np.array(
            Image.fromarray(syll).resize(resize_shape, Image.ANTIALIAS)
        )  # LANCZOS as of Pillow 2.7
    return all_syllables


def thresh_mode(syll):
    """ theshold the modal value of the syllable to ensure that the mode is 0
    """
    freqs, bins = np.histogram(syll, bins=100)
    syll = syll - bins[np.argsort(freqs)][-1]
    syll[syll < 0] = 0
    return syll


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )


def resize_compressed_syllable(syll, hparams, mel_inversion_filter=None):
    """Resizes a compressed syllable for inversion
    
    [description]
    
    Arguments:
        syll {[type]} -- [description]
        hparams {[type]} -- [description]
    """
    # remove pading around the syllable
    syll_mean = np.mean(syll, axis=0)
    if syll_mean[0] == syll_mean[-1]:
        sb = [
            i
            for i in np.arange(1, len(syll_mean))
            if (syll_mean[i] == syll_mean[0]) and (syll_mean[i - 1] == syll_mean[0])
        ]
        sb = (
            [i for i in sb if i + 1 not in sb][0],
            [i for i in sb if i - 1 not in sb][-1],
        )
        syll = syll[:, sb[0] + 1 : sb[1] - 1]

    # resizes and deconvolves with inversion filter
    if hparams["mel_filter"]:
        resize_shape = (
            int(
                (np.shape(syll)[1] / hparams["resize_samp_fr"])
                * (1000 / hparams["frame_shift_ms"])
            ),
            hparams["num_freq_final"],
        )
        syll = np.array(
            Image.fromarray(np.squeeze(syll)).resize(resize_shape, Image.ANTIALIAS)
        )
        syll = np.dot(syll.T, mel_inversion_filter).T
    else:
        resize_shape = (
            int(
                (np.shape(syll)[1] / hparams["resize_samp_fr"])
                * (1000 / hparams["frame_shift_ms"])
            ),
            hparams["num_freq"],
        )
        syll = np.array(
            Image.fromarray(np.squeeze(syll)).resize(resize_shape, Image.ANTIALIAS)
        )
    return syll


def extract_syllables(wav_loc, textgrid_file, hparams, visualize=False, verbose=False):
    """ Grabs syllable spectrograms from a wavfile given a textgrid
    """
    # read in the data
    rate, data = load_wav(wav_loc)
    hparams["sample_rate"] = rate

    # read csv
    csv_loc = wav_loc.parents[1] / "csv" / (wav_loc.stem + ".csv")
    (indv, original_wav, start_time) = pd.read_csv(csv_loc, header=None).values[0]
    start_time = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S-%f")

    # load textgrid
    textgrid_loc = wav_loc.parents[1] / "TextGrids" / (wav_loc.stem + ".TextGrid")
    try:
        textgrid = tgio.openTextgrid(fnFullPath=textgrid_loc)
    except FileNotFoundError as e:
        print(e)
        return None
    textgrid.tierDict["syllables"].entryList[:3]

    # bandpass filter data
    data = butter_bandpass_filter(
        data, hparams["lowcut"], hparams["highcut"], rate, order=2
    ).astype("float32")

    # Generates the spectrogram and also thresholds out bad spectrograms (e.g. too noisy) - take a look at wav_to_syllables.py to determine if you want this
    spec, vocal_envelope, cur_spec_thresh, fft_rate = dynamic_spectrogram(
        data / np.max(np.abs(data)),
        _mel_basis=hparams["_mel_basis"],
        hparams=hparams,
        verbose=verbose,
    )

    # get syllables
    tier = textgrid.tierDict["syllables"].entryList
    all_syllables = [
        spec[:, int(interval.start * fft_rate) : int(interval.end * fft_rate)]
        for interval in tier
    ]

    # resize syllables
    all_syllables_resized = resize_syllables(all_syllables, fft_rate, hparams)

    # set mode to 0
    all_syllables_resized = [thresh_mode(syll) for syll in all_syllables_resized]

    # 0 pad syllables
    all_syllables_comp = np.array(
        [pad_spectrogram(i, hparams["max_size_syll"]) for i in all_syllables_resized]
    )

    # convert to 8 bit
    all_syllables_comp = [
        (zero_one_norm(i) * 255).astype("uint8") for i in all_syllables_comp
    ]

    # syllable times
    syll_onsets = [interval.start for interval in tier]
    syll_lens = [interval.end - interval.start for interval in tier]
    syll_labels = [interval.label for interval in tier]

    if visualize:
        plot_syllable_list(
            all_syllables_comp, hparams["num_freq_final"], max_rows=10, width=128 * 7
        )

        plot_bout_to_syllable_pipeline(
            data,
            vocal_envelope,
            spec,
            all_syllables,
            syll_onsets,
            syll_lens,
            rate,
            fft_rate,
        )

    return (
        np.repeat(indv, len(syll_lens)),
        np.repeat(original_wav, len(syll_lens)),
        np.repeat(start_time, len(syll_lens)),
        all_syllables_comp,
        syll_onsets,
        syll_lens,
        syll_labels,
    )


from sklearn.externals.joblib import Parallel, delayed
from tqdm.autonotebook import tqdm


def prepare_syllable_dataset(
    indv_name,
    wav_list,
    dataset_loc,
    textgrid_loc,
    hparams,
    parallel=False,
    n_jobs=10,
    par_verbosity=10,
    verbose=False,
    visualize=False,
):
    """[summary]
    
    [description]
    
    Arguments:
        indv_name {[type]} -- [description]
        wav_list {[type]} -- [description]
        hparams {[type]} -- [description]
        dataset_loc {[type]} -- [description]
    
    Keyword Arguments:
        n_jobs {number} -- [description] (default: {10})
        par_verbosit {number} -- [description] (default: {10})
        verbose {bool} -- [description] (default: {False})
        visualize {bool} -- [description] (default: {False})
    """


    # loop through and get syllables
    if parallel:
        with Parallel(n_jobs=n_jobs, verbose=par_verbosity) as parallel:
            syllable_data = parallel(
                delayed(extract_syllables)(
                    wav_file,
                    textgrid_loc / (wav_file.stem + ".TextGrid"),
                    hparams,
                    visualize=visualize,
                    verbose=verbose,
                )
                for wav_file in tqdm(wav_list)
            )

    else:
        syllable_data = [
            extract_syllables(
                wav_file,
                textgrid_loc / (wav_file.stem + ".TextGrid"),
                hparams,
                visualize=visualize,
                verbose=verbose,
            )
            for wav_file in tqdm(wav_list)
        ]

    # remove any Nones from syllable data
    syllable_data = [i for i in syllable_data if i is not None]

    # the fields saved in the HDF5 file 
    key_list = (
        'indv', # ID of individual
        'original_wav_name', # location of original wav
        'wav_datetime', # datetime of the original wav
        'syllables', # spectrogram of syllables
        'syll_start_rel_wav', # time relative to bout file that this 
        'syll_lengths', # length of the syllable
        'syll_labels', # label of syllables
       ) 
    indv_data = {key: [] for key in key_list}
    # unpack/flatten data grabbed from loop
    for dtype, darray in zip(key_list, list(zip(*syllable_data))):
            [indv_data[dtype].extend(element) for element in darray] 
            indv_data[dtype] = np.array(indv_data[dtype])

    return indv_data

