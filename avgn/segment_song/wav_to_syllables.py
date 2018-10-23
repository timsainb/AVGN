import numpy as np
import os
import sys
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy import ndimage
import copy
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import h5py
from PIL import Image

import avgn.spectrogramming.spectrogramming as sg


#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skimage.transform


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        image = plt.cm.viridis(image)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)
# import local methods from the source code
from avgn.spectrogramming.make_spec import *
from avgn.segment_song.preprocessing import *


def sub_mode(syll):
    """ Subtract the mode from an array/syllable
    """
    freqs, bins = np.histogram(syll, bins=100)
    syll = syll - bins[np.argsort(freqs)][-1]  # np.median(syll)#
    syll[syll < 0] = 0
    return syll


def create_audio_envelope_waveform(waveform, wav_spectrogram, downsampled_rate,
                                   rate=44100, gauss_sigma_s=0.0,
                                   smoothing='gaussian', signal='spectrogram'
                                   ):
    """ Creates an audio envelope from waveform"""
    if signal == 'waveform':
        downsampled = scipy.signal.resample(np.abs(waveform),
                                            int(float(len(waveform))/rate*downsampled_rate))
    elif signal == 'spectrogram':
        downsampled = np.max(wav_spectrogram, axis=1)
    if smoothing == 'gaussian':
        gauss_sigma_f = downsampled_rate*gauss_sigma_s
        return norm(ndimage.filters.gaussian_filter(np.array(downsampled, np.float32), gauss_sigma_f))
    elif smoothing == 'none':
        return norm(downsampled)


def contiguous_regions(condition):
    # Compute contiguous region of silence to ensure noise levels are sufficiently low
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


def temporal_segmentation_int(vocal_envelope, fft_rate, segmentation_rate=1., threshold_max=.3):
    """ Adds dynamic threshold to vocal envelope
    """
    last_seg = 0
    seg_thresh = np.zeros((len(vocal_envelope)))
    for i in np.arange(1, len(vocal_envelope)):
        if vocal_envelope[i] > seg_thresh[i-1]:
            seg_thresh[i] = seg_thresh[i-1] + (segmentation_rate/fft_rate)
        else:
            vocal_envelope[i] = 0
        if seg_thresh[i] > threshold_max:
            seg_thresh[i] = threshold_max

    return vocal_envelope, seg_thresh


def generate_fft_time_idx(vocalization, mel_spec, rate):
    # generate time index for wav
    fft_time_idx = np.arange(0, len(vocalization), float(len(vocalization))/np.shape(mel_spec)[0])
    fft_rate = 1. / (len(vocalization)/float(rate) / float(np.shape(mel_spec)[0]))
    return fft_time_idx, fft_rate


def compute_spec_and_env(vocalization, time_index, hparams, rate, _mel_basis, subtract_median=True, mel_filter=False, verbose=False):
    """ iteratively computes a spectrogram
    """
    #for key,val in hparams.items(): exec(key + '=val')
    for k, v in hparams.items():
        globals()[k] = v
    envelope_is_good = False
    hparams_copy = copy.deepcopy(hparams)
    # loop through until conditions are met (or call this WAV bad)
    while envelope_is_good == False:

        if hparams_copy['min_level_db'] > spec_thresh_min:  # end loop if thresh is too low
            return 0, 0, 0, 0, 0
        if verbose:
            print(hparams_copy['min_level_db'], spec_thresh_min)

        if mel_filter:
            spec = norm_zero_one(sg.melspectrogram(vocalization, hparams_copy, _mel_basis))
        else:
            spec = norm_zero_one(sg.spectrogram(vocalization, hparams_copy))

        if subtract_median:
            med = np.median(spec, axis=1)
            spec = spec - np.repeat(med, np.shape(spec)[1]).reshape(np.shape(spec))
            med = np.array([bn[np.argsort(freq)][-1]
                            for freq, bn in [np.histogram(spec[i, :], bins=50) for i in range(len(spec))]])
            spec = spec - np.repeat(med, np.shape(spec)[1]).reshape(np.shape(spec))

        spec_orig = copy.deepcopy(spec)
        spec[spec < (mel_noise_filt*np.max(spec))] = 0  # filter out lowest noise

        # Grab timing into
        fft_time_idx, fft_rate = generate_fft_time_idx(vocalization, spec.T, rate)

        # Calculate a vocal envelope
        vocal_envelope = create_audio_envelope_waveform(np.abs(vocalization),
                                                        wav_spectrogram=spec.T, downsampled_rate=fft_rate,
                                                        rate=rate, gauss_sigma_s=gauss_sigma_s,
                                                        smoothing=smoothing, signal=envelope_signal
                                                        )
        vocal_envelope = np.concatenate((vocal_envelope, np.zeros(1000)))[:np.shape(spec)[1]]

        hparams_copy['min_level_db'] += spec_thresh_delta  # shift the noise threshold down

        # Look at how much silence exists in the signal
        cr = np.array(contiguous_regions(vocal_envelope == 0))

        if len(cr) == 0:
            continue

        if np.max(cr[:, 1] - cr[:, 0]) > fft_rate*min_silence_for_spec:  # needs .5 seconds of silence
            envelope_is_good = True

    spec = np.floor(norm_zero_one(spec_orig) * 255.)

    return spec, vocal_envelope, hparams['min_level_db'], fft_time_idx, fft_rate


def detect_onsets_offsets(data, threshold, min_distance):
    """
    detects when a when a signal jumps above zero, and when it goes back to zero
    """
    on = (data > threshold)  # when the data is greater than zero
    left_on = np.concatenate(([0], on), axis=0)[0:-1]
    onset = np.squeeze(np.where(on & (left_on != True)))
    offset = np.squeeze(np.where((on != True) & (left_on == True)))

    if data[-1] > threshold:
        offset = np.append(offset, len(data))  # make sure there is an offset at some point...

    if len(np.shape(onset)) < 1:
        offset = [offset]
        onset = [onset]

    new_offset = []
    new_onset = []
    if len(onset) > 0:
        new_onset.append(onset[0])
        if len(onset) > 1:
            for i in range(len(onset)-1):
                if (onset[i+1] - offset[i]) > min_distance:
                    new_onset.append(onset[i+1])
                    new_offset.append(offset[i])

        new_offset.append(offset[-1])
    return new_onset, new_offset


def cut_syllables(onsets, offsets, mel_spec, fft_time_idx, params):
    globals().update(params)
    # Cut into syllables
    all_syllables = []
    all_syllables_time_idx = []

    for i, on in enumerate(onsets):
        if offsets[i] > on+min_len:  # if the syllable is sufficiently long
            cur_seg_points = np.concatenate((
                [np.squeeze(onsets[i])], [np.squeeze(offsets[i])]))
            for j in range(len(cur_seg_points)-1):
                all_syllables.append(mel_spec[:, cur_seg_points[j]:cur_seg_points[j+1]])
                all_syllables_time_idx.append(fft_time_idx[cur_seg_points[j]:cur_seg_points[j+1]])

    syll_start = [i[0] for i in all_syllables_time_idx]  # Syllable start times, seconds
    return all_syllables, all_syllables_time_idx, syll_start

# threshold/remove bad syllables


def threshold_syllables(all_syllables, all_syllables_time_idx, syll_start, min_syll_len_s, fft_rate, power_thresh=.3, visual=False, max_vis=20):
    """ Threshold syllables based on length
    """
    # Threshold time
    good_sylls = np.array([np.shape(i)[1] for i in all_syllables])/float(fft_rate) >= min_syll_len_s

    all_syllables = [all_syllables[i] for i, b in enumerate(good_sylls) if b == True]
    all_syllables_time_idx = np.array(all_syllables_time_idx)[good_sylls]
    syll_start = np.array(syll_start)[good_sylls]

    good_sylls = np.array([np.max(i) for i in all_syllables]) >= power_thresh

    if visual == True:
        # Threshold low power
        fig, ax = plt.subplots(nrows=3, ncols=max_vis, figsize=(2*max_vis, 6))
        for syll_i, syll in enumerate(all_syllables):
            if good_sylls[syll_i] == True:
                cm = plt.cm.afmhot
            else:
                cm = plt.cm.bone
            ax[0, syll_i].matshow(syll,
                                  cmap=cm,
                                  origin='lower',
                                  aspect='auto',
                                  interpolation='nearest',
                                  vmin=0, vmax=1
                                  )
            ax[1, syll_i].plot(np.sum(syll, axis=0))
            ax[2, syll_i].plot(np.sum(syll, axis=1))
            if syll_i >= max_vis-1:
                break

        plt.show()

    all_syllables = [all_syllables[i] for i, b in enumerate(good_sylls) if b == True]
    all_syllables_time_idx = np.array(all_syllables_time_idx)[good_sylls]
    syll_start = np.array(syll_start)[good_sylls]

    return all_syllables, all_syllables_time_idx, syll_start


def plot_seg_spec(all_seg_points, mel_spec, fft_time_idx, vocal_envelope_int, seg_thresh, params, figsize=(60, 9)):
    """ Draw plot of segmentation
    """
    globals().update(params)

    exec (','.join(params) + ', = params.values()')

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    cax = ax[0].matshow(mel_spec, interpolation=None, aspect='auto',
                        origin='lower')  # cmap=plt.cm.afmhot
    for i in all_seg_points:
        ax[0].axvline(i, color='w', linestyle='-', lw=3,  alpha=.75)  # , ymax=.25)
    cax = ax[1].matshow(mel_spec, interpolation=None, aspect='auto',
                        origin='lower')  # cmap=plt.cm.afmhot
    ax[1].axhline(y=FOI_min, color='w')
    ax[1].axhline(y=FOI_max, color='w')
    ax[0].set_xlim([0, len(vocal_envelope_int)])
    ax[1].set_xlim([0, len(vocal_envelope_int)])
    ax[2].plot(norm(np.max(mel_spec, axis=0)), color='red')
    ax[2].plot(norm(vocal_envelope_int), color='k')
    ax[2].set_xlim([0, len(vocal_envelope_int)])

    plt.show()


def second_pass_threshold(onsets, offsets, vocal_envelope, new_fft_rate, params):
    """ The first pass threshold finds bouts based upon silences. The second pass, sets a more liberal threshold, based upon stretches of audio
    which are controlled for the expected bout rate

    """
    globals().update(params)
    onsets_full = []
    offsets_full = []

    for oi, (onset, offset) in enumerate(zip(onsets, offsets)):
        #print(onset, offset)
        time_len = (offset-onset)*new_fft_rate
        if time_len > ebr_max:
            # for each group of syllables that is too long
            cur_thresh = slow_threshold
            if cur_thresh < max_thresh:
                while cur_thresh < max_thresh:
                    new_onsets, new_offsets = detect_onsets_offsets(norm(vocal_envelope[onset:offset]),
                                                                    threshold=cur_thresh,
                                                                    min_distance=0.
                                                                    )
                    if cur_thresh >= max_thresh:
                        break
                     # if the number of syllables meets the expected syllabic rate
                    syllabic_rate = time_len/float(len(new_onsets) - 1)
                    if (syllabic_rate >= ebr_min) & (syllabic_rate <= ebr_max):
                        break
                    cur_thresh += thresh_delta

                onsets_full.extend(np.ndarray.flatten(np.array(new_onsets))+onset)
                offsets_full.extend(np.ndarray.flatten(np.array(new_offsets))+onset)
            else:
                onsets_full.append(onset)
                offsets_full.append(offset)
        else:
            onsets_full.append(onset)
            offsets_full.append(offset)
    onsets = onsets_full
    offsets = offsets_full
    onsets.sort()
    offsets.sort()
    return onsets, offsets


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = (pad_length - np.shape(spectrogram)[1])
    pad_left = np.floor(float(excess_needed)/2).astype('int')
    pad_right = np.ceil(float(excess_needed)/2).astype('int')
    return np.pad(spectrogram, [(0, 0), (pad_left, pad_right)], 'constant', constant_values=0)


def resize_spectrograms(all_syllables, max_size, resize_samp_fr, fft_rate, n_freq, pad_length):
    """ Resizes and pads a list of spectrograms
    """
    for i, syll in enumerate(all_syllables):
        resize_shape = [int((np.shape(syll)[1]/fft_rate)*resize_samp_fr), n_freq]
        if resize_shape[0] > pad_length:
            resize_shape[0] = pad_length
        all_syllables[i] = np.array(Image.fromarray(syll).resize(
            resize_shape, Image.ANTIALIAS))  # LANCZOS as of Pillow 2.7
    return all_syllables


def plt_all_syllables(all_syllables, n_mel_freq_components, max_rows=3, max_sylls=100, width=400, zoom=1, spacing=1, cmap=plt.cm.viridis):
    """ Plots computed syllables in a grid"""

    canvas = np.zeros((n_mel_freq_components*max_rows, width))
    x_loc = 0
    row = 0

    for i, syll in enumerate(all_syllables):

        # if the syllable is too long
        if np.shape(syll)[1] > width:
            continue

        if (x_loc+np.shape(syll)[1]) > width:
            if row == max_rows-1:
                break

            else:
                row += 1
                x_loc = 0

        canvas[row*n_mel_freq_components: (row+1)*n_mel_freq_components,
               x_loc:(x_loc+np.shape(syll)[1])] = np.flipud(syll)

        x_loc += np.shape(syll)[1] + spacing

    if row < max_rows:
        canvas = canvas[:(row+1)*n_mel_freq_components]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/32*zoom, max_rows*zoom))
    ax.matshow(canvas,
               cmap=cmap,
               # origin='lower',
               aspect='auto',
               interpolation='nearest'
               )
    plt.show()


def plot_pipeline(data, vocal_envelope, wav_spectrogram, onsets, offsets, all_syllables, rate,
                  all_syllables_time_idx, syllable_lengths, zoom=1, submode=True, figsize=(50, 10)):
    plt.clf()
    # create a plot where the top is waveform, underneath is spectrogram, underneath is segmented syllables
    #fig = plt.subplots(figsize=figsize)
    #gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,2,1])
    #ax=[plt.subplot(i) for i in gs]
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    ax[0].plot(data, color='black')
    ax[0].set_xlim([0, len(data)])
    ax[0].axis('off')
    ax[0].set_ylim([np.min(data), np.max(data)])

    ax[1].plot(vocal_envelope)
    ax[1].set_xlim([0, len(vocal_envelope)])
    ax[1].set_ylim([np.min(vocal_envelope), np.max(vocal_envelope)])

    ax[1].axis('off')

    ax[2].matshow(wav_spectrogram.T, interpolation=None, aspect='auto',  # cmap=plt.cm.gray_r,
                  cmap=plt.cm.viridis, origin='lower', extent=[0, len(data)/rate, 0, rate/2])

    for on, off in zip(np.array(onsets)/float(len(wav_spectrogram))*(len(data)/rate), np.array(offsets)/float(len(wav_spectrogram))*(len(data)/rate)):
        ax[2].axvline(on, color='white', linestyle='-', lw=1,  alpha=.75)
        ax[2].axvline(off, color='white', linestyle='-', lw=1,  alpha=.75)

    for si, syll_se in enumerate([(i[0], i[-1]) for i in all_syllables_time_idx]):
        imscatter((syll_se[1]+syll_se[0])/2, 0,
                  np.flipud(norm(all_syllables[si])), zoom=zoom, ax=ax[3])
        ax[3].text((syll_se[1]+syll_se[0])/2, .15, round(syllable_lengths[si], 3),
                   fontsize=15, horizontalalignment='center')

    ax[3].set_xlim([0, len(data)])
    ax[3].set_ylim([-0.2, 0.2])
    ax[3].axis('off')
    plt.tight_layout()
    plt.show()


def process_bout(wav_file, _mel_basis, hparams,submode=True, visualize=False):
    """ Processes a single bout for syllable information
    """
    globals().update(hparams)
    # print(wav_file)
    # load bird info
    if visualize:
        print(wav_file)
    csv_loc = '/'.join(wav_file.split('/')[:-2] + ['csv'] + [wav_file.split('/')[-1][:-4] + '.csv'])
    try:
        rate, vocalization = wavfile.read(wav_file)
    except:
        print('wav did not load')
    try:
        (bird, original_wav, start_time) = pd.read_csv(csv_loc, header=None).values[0]
    except:
        if visualize:
            print('CSV does not exist')
        return [], [], [], [], []
    start_time = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S-%f")
    # bandpass filter data
    vocalization = butter_bandpass_filter(vocalization, lowcut, highcut, rate, order=2)

    # Compute spectrogram
    spec, vocal_envelope, cur_spec_thresh, fft_time_idx, fft_rate = compute_spec_and_env(
        (vocalization/32768.).astype('float32'), start_time, hparams, rate, _mel_basis, mel_filter=mel_filter, verbose=False)

    # Return nothing if spectrogram function outputs nothing
    if np.shape(spec) == ():
        if visualize:
            print('failed making spectrogram')
        return [], [], [], [], []

    # detect the onsets and offsets of sound from the vocal envelope
    spec_to_use = norm(spec)

    # Detect onsets and offsets of vocal envelope
    # vocal_envelope = norm(np.sum(spec, axis = 0))
    onsets, offsets = detect_onsets_offsets(vocal_envelope,
                                            threshold=silence_threshold,
                                            min_distance=0.
                                            )
    # print(onsets[0:10], offsets[0:10], len(onsets))
    new_fft_rate = len(vocalization)/rate/float(np.shape(spec)[1])
    for i in range(second_pass_threshold_repeats):
        onsets, offsets = second_pass_threshold(
            onsets, offsets, vocal_envelope, new_fft_rate, hparams)

    # if the wav is empty continue (unlikely)
    if len(onsets) == 0:
        if visualize:
            print('No onsets')
        return [], [], [], [], []

    # segment into syllables based upon onset/offsets
    all_syllables, all_syllables_time_idx, syll_start = cut_syllables(
        onsets, offsets, spec, fft_time_idx, hparams)

    # threshold/remove bad syllables
    all_syllables, all_syllables_time_idx, syll_start = threshold_syllables(
        all_syllables, all_syllables_time_idx, syll_start, min_syll_len_s, fft_rate, power_thresh=power_thresh)

    # Threshold for too few syllables
    if len(all_syllables) < min_num_sylls:
        if visualize:
            print('Not enough syllables')
        return [], [], [], [], []

    # get syllable timing information
    syll_start_dt = [dati.strftime("%d/%m/%y %H:%M:%S.%f") for dati in [start_time + timedelta(
        seconds=st/rate) for st in syll_start]]  # datetime info for when the syllable occurs
    # length of each syllable
    syllable_lengths = [np.shape(all_syllables[i])[1]/fft_rate for i in range(len(all_syllables))]

    # resize spectrogram
    all_syllables = resize_spectrograms(all_syllables, max_size=max_size_syll, resize_samp_fr=resize_samp_fr,
                                        fft_rate=fft_rate, n_freq=num_freq_final, pad_length=pad_length)
    # set mode to 0
    if submode:
        all_syllables = [sub_mode(syll) for syll in all_syllables]

    # 0 pade
    all_syllables = np.array([pad_spectrogram(i, pad_length) for i in all_syllables])
    all_syllables = [(norm(i)*255).astype('uint8') for i in all_syllables]

    if visualize == True:
        print('spec thresh: ', cur_spec_thresh, 'spec mean: ', np.mean(spec))
        plot_pipeline(vocalization, vocal_envelope, spec.T, onsets, offsets, all_syllables,
                      rate, all_syllables_time_idx, syllable_lengths, zoom=0.4, figsize=(50, 5))
        #print(('Memory usage'+ str( process.memory_info().rss*1e-9)))

    # add to the wav
    wav_file = np.array([wav_file for i in range(len(all_syllables))])

    return wav_file, all_syllables, syll_start_dt, syll_start, syllable_lengths


dt = h5py.special_dtype(vlen=str)


def save_dataset(location, all_bird_syll, starting_times, lengths, wav_file, syll_start_rel_to_wav, bird_name):
    """
    saves dataset as an hdf5
    """
    print('Saving dataset to ', location)
    with h5py.File(location, 'w') as f:
        f.attrs['bird_name'] = bird_name
        dset_spec = f.create_dataset("spectrograms", np.shape(
            all_bird_syll), dtype='uint8', data=all_bird_syll)
        dset_start = f.create_dataset("start", (len(starting_times), 1),
                                      dtype=dt, data=starting_times)
        dset_wav_file = f.create_dataset("wav_file", (len(wav_file), 1), dtype=dt, data=wav_file)
        dset_syll_start_rel_to_wav = f.create_dataset("syll_start_rel_wav",
                                                      np.shape(syll_start_rel_to_wav), dtype='float32', data=syll_start_rel_to_wav)
        dset_lengths = f.create_dataset("lengths", np.shape(lengths), dtype='float32', data=lengths)


def iterateCreateSyllSpec(data, _mel_basis, hparams, pct_fail, power_thresh):
    """
    dynamic thresholding to keep noisy wavs the same noise level
    """

    wav_spectrogram = None
    hparams_copy = copy.deepcopy(hparams)
    cur_spec_thresh = hparams_copy['min_level_db']
    thresh_min = hparams_copy['spec_thresh_min']
    while (cur_spec_thresh < thresh_min) & (wav_spectrogram is None):
        hparams_copy['min_level_db'] = cur_spec_thresh
        wav_spectrogram = norm_zero_one(sg.melspectrogram(data, hparams_copy, _mel_basis))
        wav_spectrogram[wav_spectrogram < power_thresh] = 0
        pct_sil = np.sum(wav_spectrogram < power_thresh)/np.prod(np.shape(wav_spectrogram))
        # plt.matshow(wav_spectrogram);plt.show();print(pct_sil)
        if pct_sil < pct_fail:
            cur_spec_thresh += hparams_copy['spec_thresh_delta']
            wav_spectrogram = None
    return wav_spectrogram


def getSyllsFromWav(row, _mel_basis, wav_time, hparams):
    """ Extract syllables from wav file
    """
    globals().update(hparams)
    try:
        rate, data = wavfile.read(row.WavLoc)
    except:
        print('WAV file did not load: ' + row.WavLoc)
        return None
    data = butter_bandpass_filter(
        data, hparams['lowcut'], hparams['highcut'], rate, order=3).astype('int16')
    syll_info = []
    # go through tier
    for (syll_start, syll_len, syll_sym) in zip(row.NotePositions,
                                                row.NoteLengths,
                                                row.NoteLabels):
        syll_stop = syll_start + syll_len

        # if there are labelled syllables past the wav length
        if len(data[syll_start:syll_stop]) <= 0:
            print('Mismatched time in textgrid/wav ' + row.WavLoc)
            return None

        # extract the syllable from the song
        mel_spec = iterateCreateSyllSpec(data=data[syll_start:syll_stop].astype(
            'float64'), _mel_basis=_mel_basis, hparams=hparams, pct_fail=hparams['pct_fail'], power_thresh=hparams['power_thresh'])
        if mel_spec is None:
            syll_info.append(None)
            continue
        mel_spec[mel_spec == np.median(mel_spec)] = 0

        # pad and resize syllable
        fft_rate = np.shape(mel_spec)[1]/(syll_stop/rate - syll_start/rate)
        mel_spec = resize_spectrograms([mel_spec], max_size=max_size_syll,
                                       resize_samp_fr=resize_samp_fr, fft_rate=fft_rate,
                                       n_freq=num_freq_final, pad_length=pad_length)
        mel_spec = pad_spectrogram(mel_spec[0], pad_length)

        mel_spec[mel_spec < 0] = 0
        mel_spec = ((mel_spec/np.max(mel_spec))*255).round().astype('uint8')

        syll_info.append([row.WavLoc, mel_spec, (
            wav_time + timedelta(seconds=syll_start/rate)).strftime("%d/%m/%y %H:%M:%S.%f"),
            syll_start,
            (syll_stop - syll_start)/rate, syll_sym]
        )
    return syll_info
