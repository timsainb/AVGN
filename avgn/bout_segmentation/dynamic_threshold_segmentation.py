import numpy as np
import copy
import avgn.signalprocessing.spectrogramming as sg
from avgn.utils.general import zero_one_norm
from tqdm.autonotebook import tqdm
from praatio import tgio
from avgn.utils.audio import load_wav
import pandas as pd
from datetime import datetime
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.visualization.spectrogram import (
    plot_segmentations,
)

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
    if hparams["mel_filter"]:
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
        if verbose:
            print("Current min_level_dB: {}".format(hparams_current["min_level_dB"]))

    if not envelope_is_good:
        return None, None, None
    else:
        return spec, vocal_envelope, hparams_current["min_level_dB"], fft_rate


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


def boundaries_to_textgrid(syllable_start, syllable_lens, save_loc=None):
    """ create a textgrid from start/stop times
    """
    tg = tgio.Textgrid()

    syllablesTier = tgio.IntervalTier(
        "syllables",
        [
            (syll_start, syll_start + syll_len, "syll")
            for syll_start, syll_len in zip(syllable_start, syllable_lens)
        ],
    )
    tg.addTier(syllablesTier)
    
    # save
    if save_loc is not None:
        tg.save(save_loc, minimumIntervalLength=None)

    return tg

def textgrid_from_wav(wav_loc, textgrid_loc, hparams, visualize = False, verbose=False):
    # load the wav
    rate, data = load_wav(wav_loc)
    hparams["sample_rate"] = rate

    # load the csv with datetime info
    csv_loc = wav_loc.parents[1]/'csv'/(wav_loc.stem + '.csv')
    (bird, original_wav, start_time) = pd.read_csv(csv_loc, header=None).values[0]
    start_time = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S-%f")

    # bandpass filter data
    data = butter_bandpass_filter(
        data, hparams["lowcut"], hparams["highcut"], rate, order=2
    ).astype("float32")

    # bout statistics
    if verbose:
        print(
            "Rate: ",
            rate,
            "Time sung: ",
            start_time,
            "Length:",
            len(data) / float(rate),
        )

    # Generates the spectrogram and also thresholds out bad spectrograms 
    #   (e.g. too noisy) - take a look at wav_to_syllables.py to determine if you want this
    spec, vocal_envelope, cur_spec_thresh, fft_rate = dynamic_spectrogram(
        data / np.max(np.abs(data)), _mel_basis=None, hparams=hparams, verbose=verbose
    )

    # Detect onsets and offsets of vocal envelope
    onsets, offsets = np.array(
        contiguous_regions(vocal_envelope > hparams["silence_threshold"])
    ).T

    # segment into syllables based upon onset/offsets
    all_syllables, all_syllable_starts, all_syllable_lens = cut_syllables(
        onsets, offsets, spec, fft_rate, hparams
    )

    # threshold syllables
    syll_len_mask = np.array(all_syllable_lens) >= hparams["min_syll_len_s"]
    all_syllables = [syll for syll, mask in zip(all_syllables, syll_len_mask) if mask]
    all_syllable_starts = np.array(all_syllable_starts)[syll_len_mask]
    all_syllable_lens = np.array(all_syllable_lens)[syll_len_mask]

    # if there are only a few syllables, this isn't really a bout
    if len(all_syllables) < hparams['min_num_sylls']:
        return
    # create the textgrid
    boundaries_to_textgrid(all_syllable_starts, all_syllable_lens, textgrid_loc)

    if visualize:
        # plot again only with true syllables
        plot_segmentations(
            spec, vocal_envelope, all_syllable_starts, all_syllable_lens, fft_rate, hparams
        )

