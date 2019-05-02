from scipy.io import wavfile
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from datetime import timedelta

# for signalling timeouts
import signal

from avgn.utils.timeouts import TimeoutException
from avgn.signalprocessing.filtering import RMS, butter_bandpass_filter
from avgn.signalprocessing.onsets import detect_onsets_offsets


def compute_spectral_features(
    data, wav_spectrogram, vfrmin, vfrmax, noise_thresh, freq_step_size
):
    """Computes features of a spectrogram
    """
    norm_power = np.mean(wav_spectrogram, axis=0)
    norm_power = (norm_power - np.min(norm_power)) / (
        np.max(norm_power) - np.min(norm_power)
    )
    vocal_power_ratio = np.mean(norm_power[vfrmin:vfrmax]) / np.mean(
        norm_power[vfrmax:]
    )
    center_of_mass_f = (
        vfrmin + ndimage.measurements.center_of_mass(norm_power[vfrmin:])[0]
    ) * freq_step_size
    vrp = np.sum(wav_spectrogram[:, vfrmin:vfrmax], axis=1)
    vrp = (vrp - np.min(vrp)) / (np.max(vrp) - np.min(vrp))
    pct_silent = np.sum(vrp <= noise_thresh) / float(len(vrp))
    max_power_f = np.argmax(norm_power) * freq_step_size
    max_amp = np.max(data)
    return (
        norm_power,
        vocal_power_ratio,
        center_of_mass_f,
        vrp,
        pct_silent,
        max_power_f,
        max_amp,
    )


def load_wav(wav_loc):
    try:
        rate, data = wavfile.read(wav_loc)
        return rate, data
    except:
        print("failed to load %s" % (wav_loc))
        return None, None


def int16_to_float32(data):
    """ Converts from uint16 wav to float32 wav
    """
    return


def float32_to_uint8(data):
    """ convert from float32 to uint8 (256)
    """
    return


def float32_to_int16(data):
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    return np.array(data * 32767).astype("int16")


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
    rate, data = wavfile.read(wav_info)
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

    # threshold clips of sound
    for onset_sound, offset_sound in zip(onset_sounds, offset_sounds):
        clip = threshold_clip(data[onset_sound:offset_sound], rate, params)
        return clip


def threshold_clip(clip, rate, params):
        # get the length of the segment
        segment_length = len(clip) / float(rate)
        
        # return if the clip is the wrong length
        if segment_length <= params['min_segment_length_s']:
            return
        if segment_length >= params['max_segment_length_s']:
            return







"""def process_bird_wav(
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
    for k, v in params.items():
        globals()[k] = v
    signal.alarm(seconds_timeout)  # timeout on a WAV file if it takes too long
    try:
        exec(",".join(params) + ", = params.values()")

        # Load up the WAV
        try:
            rate, data = wavfile.read(wav_info)
        except:
            print("failed to load %s" % (wav_info))
            return

        freq_step_size = (rate / 2) / (
            num_freq
        )  # corresponds to frequency bins of spectrogram

        vfrmin = int(vocal_freq_min / freq_step_size)
        vfrmax = int(vocal_freq_max / freq_step_size)

        # we only want one channel
        if len(np.shape(data)) == 2:
            data = data[:, 0]

        # Bandpass filter the data
        data = butter_bandpass_filter(
            data.astype("float32"), lowcut, highcut, rate, order=2
        )  # .astype('int16')
        data = np.array(data / np.max(np.abs(data)) * 32767).astype("int16")

        # threshold the audio
        rms_data, sound_threshed = RMS(
            data, rms_stride, rms_window, rms_padding, rate, noise_thresh
        )

        # Find the onsets/offsets of sound
        onset_sounds, offset_sounds = detect_onsets_offsets(
            np.repeat(sound_threshed, int(rms_stride * rate)),
            threshold=0,
            min_distance=0,
        )

        # make a list of your sounds
        for onset_sound, offset_sound in zip(onset_sounds, offset_sounds):
            segment_length = (offset_sound - onset_sound) / float(rate)
            if (
                segment_length >= min_segment_length_s
                and segment_length <= max_segment_length_s
            ):
                offset_sound = np.squeeze(offset_sound)  # why does this need to exist??

                # Filtering
                if np.max(data[onset_sound:offset_sound]) < min_amp_val:
                    # if verbose:print ('Skipping for amplitude')
                    # color = 'blue'
                    continue

                # Build a waveform to threshold out 'bad spectrograms'
                onset_sound = int(onset_sound - rate * segment_padding)
                offset_sound = int(offset_sound + rate * segment_padding)
                if onset_sound < 0:
                    onset_sound = 0
                if offset_sound >= len(data):
                    offset_sound = len(data) - 1

                wav_spectrogram = sg.spectrogram(
                    (data[onset_sound:offset_sound] / 32768.0).astype("float32"), params
                ).T

                # Compute thresholding statistics
                norm_power, vocal_power_ratio, center_of_mass_f, vrp, pct_silent, max_power_f, max_amp = compute_spectral_features(
                    data[onset_sound:offset_sound],
                    wav_spectrogram,
                    vfrmin,
                    vfrmax,
                    noise_thresh,
                    freq_step_size,
                )

                # run thresholds
                if max_power_f < max_power_f_min:
                    if verbose:
                        print("Skipping for MAX_POWER_F", max_power_f, max_power_f_min)
                    # color = 'blue'
                    continue
                if pct_silent < min_silence_pct:
                    if verbose:
                        print("Skipping for PCT_SILENT")
                    # color = 'blue'
                    continue

                start_time = wav_time + timedelta(seconds=onset_sound / float(rate))
                time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S-%f")

                # visualization
                if visualize:
                    color = "red"
                    print("vocal_power_ratio: ", vocal_power_ratio)
                    print(
                        "max_power(Hz): ",
                        np.argmax(np.mean(wav_spectrogram, axis=0))
                        * (rate / (num_freq / 2)),
                    )
                    print("max_amp: ", np.max(data[onset_sound:offset_sound]))
                    print("center_of_mass_f", center_of_mass_f)
                    print("segment_length(s): ", segment_length)
                    print("pct_silent(%): ", pct_silent)
                    print("max_power_f: ", max_power_f)
                    print("onset_t_rel_wav", onset_sound / rate)

                    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 4))
                    if color == "blue":
                        ax[0].matshow(
                            wav_spectrogram.T,
                            interpolation=None,
                            aspect="auto",
                            cmap=plt.cm.bone,
                            origin="lower",
                        )
                    else:
                        ax[0].matshow(
                            wav_spectrogram.T,
                            interpolation=None,
                            aspect="auto",
                            cmap=plt.cm.afmhot,
                            origin="lower",
                        )

                    ax[1].plot(norm_power)
                    ax[1].axvspan(vfrmin, vfrmax, facecolor="0.5", alpha=0.5)
                    ax[1].axvline(center_of_mass_f / freq_step_size, color="red")
                    ax[1].axvline(max_power_f / freq_step_size, color="blue")
                    ax[2].plot(vrp)
                    ax[2].set_xlim((0, len(vrp)))
                    ax[1].set_xlim((0, len(norm_power)))
                    ax[0].set_title("Spectrogram")
                    ax[0].set_title(np.max(data[onset_sound:offset_sound]))
                    ax[1].set_title("Frequency by power")
                    ax[2].set_title("Time by power")
                    ax[1].set_ylabel("Power")
                    ax[2].set_ylabel("Power")
                    ax[1].set_xlabel("Frequency")
                    ax[2].set_xlabel("Time")
                    # plt.tight_layout()
                    plt.show()

                # Save wav file / origin / time / rate
                save_bout_wav(
                    data[onset_sound:offset_sound],
                    rate,
                    bird,
                    save_to_folder,
                    time_string,
                    wav_info,
                    wav_spectrogram,
                    skip_created,
                    save_spectrograms,
                )

    except TimeoutException:
        print("WAV " + wav_info + " took too long to process")
        return"""


def save_bout_wav(
    data,
    rate,
    bird,
    save_to_folder,
    time_string,
    wav_info,
    wav_spectrogram,
    skip_created=False,
    save_spectrograms=True,
):

    bird_folder = save_to_folder + bird + "/"

    # skip this wav if it already exists
    if skip_created and os.path.isfile(bird_folder + "wavs/" + time_string + ".wav"):
        return

    # Save the wav
    wavfile.write(bird_folder + "wavs/" + time_string + ".wav", rate, data)
    # bird name, start time, origin_wav
    with open(bird_folder + "csv/" + time_string + ".csv", "w") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([bird, wav_info, time_string])

    # save spectrogram image
    if save_spectrograms:
        if not os.path.exists(save_to_folder + bird + "/specs/"):
            try:
                os.makedirs(save_to_folder + bird + "/specs/")
            except:
                print("Problem making: " + save_to_folder + bird + "/specs/")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        ax.matshow(
            wav_spectrogram.T,
            interpolation=None,
            aspect="auto",
            cmap=plt.cm.afmhot,
            origin="lower",
        )

        plt.savefig(
            save_to_folder + bird + "/specs/" + time_string + ".jpg",
            bbox_inches="tight",
        )
        plt.close()
