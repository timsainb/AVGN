from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, "valid"))


def RMS(data, rate, rms_stride, rms_window, rms_padding, noise_thresh):
    """
    Take data, run and RMS filter over it
    """

    # we compute root mean squared over a window, where we stride by rms_stride seconds for speed
    rms_data = window_rms(
        data.astype("float32")[:: int(rms_stride * rate)],
        int(rate * rms_window * rms_stride),
    )
    rms_data = rms_data / np.max(rms_data)

    # convolve a block filter over RMS, then threshold it, so to call everything with RMS > noise_threshold noise
    block_filter = np.ones(int(rms_padding * rms_stride * rate))  # create our filter

    # pad the data to be filtered
    rms_threshed = np.concatenate(
        (
            np.zeros(int(len(block_filter) / 2)),
            np.array(rms_data > noise_thresh),
            np.zeros(int(len(block_filter) / 2)),
        )
    )
    # convolve on our filter
    sound_threshed = np.array(np.convolve(rms_threshed, block_filter, "valid") > 0)[
        : len(rms_data)
    ]  

    return rms_data, sound_threshed