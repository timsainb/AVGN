import numpy as np
from datetime import timedelta
from pathlib2 import Path
import csv
import os

# for signalling timeouts
# import signal
# from avgn.utils.timeouts import TimeoutException

from avgn.signalprocessing.filtering import RMS, butter_bandpass_filter
from avgn.signalprocessing.onsets import detect_onsets_offsets
from avgn.signalprocessing.spectrogramming import spectrogram
from avgn.utils.audio import load_wav, float32_to_int16, int16_to_float32, write_wav
from avgn.utils.paths import ensure_dir
from avgn.utils.general import zero_one_norm
from avgn.visualization.spectrogram import visualize_spec

##TODO: timeout dectorator
def process_bird_wav(
    bird,
    wav_info,
    wav_time,
    params,
    save_to_folder,
    visualize=False,
    skip_created=False,
    seconds_timeout=300,
    save_spectrograms=True,
    verbose=False,
):
    """splits a wav file into periods of silence and periods of sound based on params
    """
    # Load up the WAV
    rate, data = load_wav(wav_info)
    params["sample_rate"] = rate
    if rate is None or data is None:
        return

    # bandpass filter
    data = butter_bandpass_filter(
        data.astype("float32"), params["lowcut"], params["highcut"], rate, order=2
    )
    data = float32_to_int16(data)

    # we only want one channel
    if len(np.shape(data)) == 2:
        data = data[:, 0]

    # threshold the (root mean squared of the) audio
    rms_data, sound_threshed = RMS(
        data,
        rate,
        params["rms_stride"],
        params["rms_window"],
        params["rms_padding"],
        params["noise_thresh"],
    )
    # Find the onsets/offsets of sound
    onset_sounds, offset_sounds = detect_onsets_offsets(
        np.repeat(sound_threshed, int(params["rms_stride"] * rate)),
        threshold=0,
        min_distance=0,
    )
    # make sure all onset sounds are at least zero (due to downsampling in RMS)
    onset_sounds[onset_sounds < 0] = 0

    # threshold clips of sound
    for onset_sound, offset_sound in zip(onset_sounds, offset_sounds):

        # segment the clip
        clip = data[onset_sound:offset_sound]
        ### if the clip is thresholded, as noise, do not save it into dataset
        # bin width in Hz of spectrogram
        freq_step_size_Hz = (rate / 2) / params["num_freq"]
        bout_spec = threshold_clip(clip, rate, freq_step_size_Hz, params, visualize=visualize, verbose=verbose)
        if bout_spec is None:
            # visualize spectrogram if desired
            if visualize:
                # compute spectrogram of clip
                wav_spectrogram = spectrogram(int16_to_float32(clip), params)
                visualize_spec(wav_spectrogram, show=True)
            continue


        # determine the datetime of this clip
        start_time = wav_time + timedelta(seconds=onset_sound / float(rate))
        time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S-%f")

        # create a subfolder for the individual bird if it doesn't already exist
        bird_folder = Path(save_to_folder).resolve() / bird
        ensure_dir(bird_folder)

        # save data
        save_bout_wav(
            data, rate, bird_folder, bird, wav_info, time_string, skip_created
        )

        # save the spectrogram of the data
        if save_spectrograms:
            save_bout_spec(bird_folder, bout_spec, time_string, skip_created)


def save_bout_wav(
    data, rate, bird_folder, bird, orig_wav, time_string, skip_created=False
):
    """ Save the wav and a csv of the extracted bout
        
    Arguments:
        data {[type]} -- [description]
        rate {[type]} -- [description]
        save_to_folder {[type]} -- [description]
        bird {[type]} -- [description]
        orig_wav {[type]} -- [description]
        time_string {[type]} -- [description]
    
    Keyword Arguments:
        skip_created {bool} -- [description] (default: {False})
    """

    # save the wav file
    wav_folder = bird_folder / "wavs" 
    ensure_dir(wav_folder)
    wav_loc = wav_folder / (time_string + ".wav")
    # if the file already exists and skip created flag is true, dont overwrite
    if skip_created and os.path.isfile(wav_loc):
        return
    write_wav(wav_loc, rate, data)

    # write to a csv with bird, original wav location, datetime of bout
    csv_folder = bird_folder / "csv" 
    csv_loc = csv_folder / (time_string + ".csv")
    ensure_dir(csv_folder)
    with open(csv_loc, "w") as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow([bird, orig_wav, time_string])
    return


def save_bout_spec(
    bird_folder, wav_spectrogram, time_string, skip_created=False, figsize=(20, 4)
):

    # save the spec file
    spec_folder = bird_folder / "specs"
    ensure_dir(spec_folder)
    spec_loc = spec_folder / (time_string + ".jpg")
    if skip_created and os.path.isfile(spec_loc):
        return
    # plot
    visualize_spec(wav_spectrogram.T, save_loc=spec_loc, show=False, figsize=(20,5))
    return


def threshold_clip(clip, rate, freq_step_size_Hz, params, visualize=False, verbose=False):
    """ determines if a clip is a bout, or noise based on threshold parameters
    """
    # get the length of the segment
    segment_length = len(clip) / float(rate)

    # return if the clip is the wrong length
    if segment_length <= params["min_segment_length_s"]:
        if verbose:
            print('Segment length {} less than minimum of {}'.format(segment_length, params["min_segment_length_s"]))
        return
    if segment_length >= params["max_segment_length_s"]:
        if verbose:
            print('Segment length {} greather than maximum of {}'.format(segment_length, params["max_segment_length_s"]))
        return

    # compute spectrogram of clip
    wav_spectrogram = spectrogram(int16_to_float32(clip), params)
    # determine the power of the spectral envelope
    norm_power = np.mean(wav_spectrogram, axis=0)
    norm_power = (norm_power - np.min(norm_power)) / (
        np.max(norm_power) - np.min(norm_power)
    )

    # get the maximum power region of the frequency envelope
    peak_power_Hz = np.argmax(norm_power) * freq_step_size_Hz

    # threshold for the location of peak power
    if peak_power_Hz < params["vocal_range_Hz"][0]:
        if verbose:
            print('Peak power {} Hz less than minimum of {}'.format(peak_power_Hz, params["vocal_range_Hz"][0]))
        return

    # threshold based on silence
    vocal_power = zero_one_norm(
        np.sum(
            wav_spectrogram[
                :,
                int(params["vocal_range_Hz"][0] / freq_step_size_Hz) : int(
                    params["vocal_range_Hz"][1] / freq_step_size_Hz
                ),
            ],
            axis=1,
        )
    )
    # the percent of the spectrogram below the noise threshold
    pct_silent = np.sum(vocal_power <= params["noise_thresh"]) / float(len(vocal_power))
    if pct_silent < params["min_silence_pct"]:
        if verbose:
            print('Percent silent {} /% less than maximum of {}'.format(pct_silent, params["min_silence_pct"]))
        return

    if visualize:
        visualize_spec(wav_spectrogram, show=True)

    # compute threshold statistics
    return wav_spectrogram
