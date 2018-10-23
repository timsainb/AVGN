import librosa
import librosa.filters
import numpy as np

from scipy import signal
from scipy.signal import butter, lfilter

def spectrogram(y, hparams):
  D = _stft(preemphasis(y,hparams), hparams)
  S = _amp_to_db(np.abs(D)) - hparams['ref_level_db']
  return _normalize(S, hparams)

def inv_spectrogram(spectrogram, hparams):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams['power'], hparams), hparams)          # Reconstruct phase


def preemphasis(x,hparams):
  return signal.lfilter([1, -hparams['preemphasis']], [1], x)

def inv_preemphasis(x, hparams):
  return signal.lfilter([1], [1, -hparams['preemphasis']], x)

def melspectrogram(y,hparams,_mel_basis):
  D = _stft(preemphasis(y, hparams), hparams)
  S = _amp_to_db(_linear_to_mel(np.abs(D),_mel_basis)) - hparams['ref_level_db']
  return _normalize(S, hparams)


def find_endpoint(wav, hparams, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams['sample_rate'] * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def _griffin_lim(S, hparams):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles,hparams)
  for i in range(hparams['griffin_lim_iters']):
    angles = np.exp(1j * np.angle(_stft(y, hparams)))
    y = _istft(S_complex * angles, hparams)
  return y

def _stft(y, hparams):
  n_fft, hop_length, win_length = _stft_parameters(hparams)
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hparams):
  _, hop_length, win_length = _stft_parameters(hparams)
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals, hparams):
  n_fft, hop_length, win_length = _stft_parameters(hparams)
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts, hparams):
  n_fft, hop_length, win_length = _stft_parameters(hparams)
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters(hparams):
  n_fft = (hparams['num_freq'] - 1) * 2
  hop_length = int(hparams['frame_shift_ms'] / 1000 * hparams['sample_rate'])
  win_length = int(hparams['frame_length_ms'] / 1000 * hparams['sample_rate'])
  return n_fft, hop_length, win_length

def _linear_to_mel(spectrogram, _mel_basis):
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(hparams):
  n_fft = (hparams['num_freq'] - 1) * 2
  return librosa.filters.mel(hparams['sample_rate'], n_fft, n_mels=hparams['num_mels'], fmin = hparams['fmin'], fmax=hparams['fmax'])

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S, hparams):
  return np.clip((S - hparams['min_level_db']) / -hparams['min_level_db'], 0, 1)

def _denormalize(S, hparams):
  return (np.clip(S, 0, 1) * -hparams['min_level_db']) + hparams['min_level_db']

def _denormalize_tensorflow(S, hparams):
  return (tf.clip_by_value(S, 0, 1) * -hparams['min_level_db']) + hparams['min_level_db']
