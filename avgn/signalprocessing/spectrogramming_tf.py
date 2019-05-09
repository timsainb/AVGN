import tensorflow as tf

# https://github.com/keithito/tacotron/blob/master/util/audio.py
def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize_tensorflow(S, hparams):
    return tf.clip_by_value((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _amp_to_db_tensorflow(x):
    return 20 * tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))


def spectrogram_tensorflow(y, hparams):
    D = _stft_tensorflow(y, hparams)
    S = _amp_to_db_tensorflow(tf.abs(D)) - hparams.ref_level_db
    return _normalize_tensorflow(S, hparams)


def _stft_tensorflow(signals, hparams):
    return tf.signal.stft(
        signals, hparams.win_length, hparams.hop_length, hparams.n_fft, pad_end=True
    )