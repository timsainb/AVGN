import numpy as np
import copy
import avgn.signalprocessing.spectrogramming as sg
from avgn.utils.general import zero_one_norm
from tqdm.autonotebook import tqdm
from PIL import Image  # for resizing syllables


def contiguous_regions(condition):
    """
    Compute contiguous region of binary value (e.g. silence in waveform) to ensure noise levels are sufficiently low
    """
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append([x1, x2])
        i = x2
    return idx


def dynamic_spectrogram(
    vocalization,
    hparams,
    _mel_basis=None,
    subtract_freq_channel_median=True,
    verbose=False,
):
    """computes a spectrogram from a waveform by iterating through thresholds to ensure a consistent noise level
        
        Arguments:
            vocalization {[type]} -- [description]
            rate {[type]} -- [description]
            hparams {[type]} -- [description]
             {[type]} -- [description]
        
        Keyword Arguments:
            _mel_basis {[type]} -- [description] (default: {None})
            subtract_freq_channel_median {bool} -- [description] (default: {True})
            mel_filter {bool} -- [description] (default: {False})
            verbose {bool} -- [description] (default: {False})
    """

    # does the envelope meet the standards necessary to consider this a bout
    envelope_is_good = False

    # make a copy of the hyperparameters
    hparams_current = copy.deepcopy(hparams)

    # spectrogram data
    if hparams['mel_filter']:
        spec_orig = sg.melspectrogram_nn(vocalization, hparams, _mel_basis)
    else:
        spec_orig = sg.spectrogram_nn(vocalization, hparams)

    # loop through possible thresholding configurations starting at the highest
    for loopi, mldb in enumerate(
        tqdm(
            np.arange(
                hparams["min_level_dB"],
                hparams["min_level_dB_floor"],
                hparams["spec_thresh_delta_dB"],
            ),
            leave=False,
            disable=(not verbose),
        )
    ):
        # set the minimum dB threshold
        hparams_current["min_level_dB"] = mldb
        # normalize the spectrogram
        spec = zero_one_norm(sg._normalize(spec_orig, hparams_current))

        # subtract the median
        if subtract_freq_channel_median:
            spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
            spec[spec < 0] = 0

        # get the vocal envelope
        vocal_envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        # normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        # Look at how much silence exists in the signal
        cr_on = np.array(
            contiguous_regions(vocal_envelope <= hparams["silence_threshold"])
        )
        cr_off = np.array(
            contiguous_regions(vocal_envelope > hparams["silence_threshold"])  # / 10)
        )

        # if there is a silence of at least min_silence_for_spec length,
        #  and a vocalization of no greater than max_vocal_for_spec length, the env is good
        if len(cr_on) > 0:
            # frames per second of spectrogram
            fft_rate = 1000 / hparams["frame_shift_ms"]
            # longest silences and periods of vocalization
            max_silence_len = np.max(cr_on[:, 1] - cr_on[:, 0]) / fft_rate
            max_vocalization_len = np.max(cr_off[:, 1] - cr_off[:, 0]) / fft_rate
            if verbose:
                print("longest silence", max_silence_len)
                print("longest vocalization", max_vocalization_len)

            if max_silence_len > hparams["min_silence_for_spec"]:
                if max_vocalization_len < hparams["max_vocal_for_spec"]:
                    envelope_is_good = True
                    break
        hparams_current["min_level_dB"] += hparams[
            "spec_thresh_delta_dB"
        ]  # shift the noise threshold down

    if not envelope_is_good:
        return None, None, None
    else:
        return spec, vocal_envelope, hparams["min_level_dB"], fft_rate


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
    """Resizes a compressed syllable for 
    
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
        sb = [i for i in sb if i + 1 not in sb][0], [i for i in sb if i - 1 not in sb][-1]
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