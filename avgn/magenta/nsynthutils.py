# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for NSynth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import librosa
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf


def shell_path(path):
  return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


#===============================================================================
# WaveNet Functions
#===============================================================================


def load_audio(path, sample_length=64000, sr=16000):
  """Loading of a wave file.
  Args:
    path: Location of a wave file to load.
    sample_length: The truncated total length of the final wave file.
    sr: Samples per a second.
  Returns:
    out: The audio in samples from -1.0 to 1.0
  """
  audio, _ = librosa.load(path, sr=sr)
  audio = audio[:sample_length]
  return audio


def mu_law(x, mu=255, int8=False):
  """A TF implementation of Mu-Law encoding.
  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.
  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.floor(out * 128)
  if int8:
    out = tf.cast(out, tf.int8)
  return out


def inv_mu_law(x, mu=255):
  """A TF implementation of inverse Mu-Law.
  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.
  Returns:
    out: The decoded data.
  """
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out)
  return out


def inv_mu_law_numpy(x, mu=255.0):
  """A numpy implementation of inverse Mu-Law.
  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.
  Returns:
    out: The decoded data.
  """
  x = np.array(x).astype(np.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
  out = np.where(np.equal(x, 0), x, out)
  return out


def trim_for_encoding(wav_data, sample_length, hop_length=512):
  """Make sure audio is a even multiple of hop_size.
  Args:
    wav_data: 1-D or 2-D array of floats.
    sample_length: Max length of audio data.
    hop_length: Pooling size of WaveNet autoencoder.
  Returns:
    wav_data: Trimmed array.
    sample_length: Length of trimmed array.
  """
  if wav_data.ndim == 1:
    # Max sample length is the data length
    if sample_length > wav_data.size:
      sample_length = wav_data.size
    # Multiple of hop_length
    sample_length = (sample_length // hop_length) * hop_length
    # Trim
    wav_data = wav_data[:sample_length]
  # Assume all examples are the same length
  elif wav_data.ndim == 2:
    # Max sample length is the data length
    if sample_length > wav_data[0].size:
      sample_length = wav_data[0].size
    # Multiple of hop_length
    sample_length = (sample_length // hop_length) * hop_length
    # Trim
    wav_data = wav_data[:, :sample_length]

  return wav_data, sample_length


#===============================================================================
# Baseline Functions
#===============================================================================
#---------------------------------------------------
# Pre/Post-processing
#---------------------------------------------------
def get_optimizer(learning_rate, hparams):
  """Get the tf.train.Optimizer for this optimizer string.
  Args:
    learning_rate: The learning_rate tensor.
    hparams: tf.contrib.training.HParams object with the optimizer and
        momentum values.
  Returns:
    optimizer: The tf.train.Optimizer based on the optimizer string.
  """
  return {
      "rmsprop":
          tf.RMSPropOptimizer(
              learning_rate,
              decay=0.95,
              momentum=hparams.momentum,
              epsilon=1e-4),
      "adam":
          tf.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8),
      "adagrad":
          tf.AdagradOptimizer(learning_rate, initial_accumulator_value=1.0),
      "mom":
          tf.MomentumOptimizer(learning_rate, momentum=hparams.momentum),
      "sgd":
          tf.GradientDescentOptimizer(learning_rate)
  }.get(hparams.optimizer)


def specgram(audio,
             n_fft=512,
             hop_length=None,
             mask=True,
             log_mag=True,
             re_im=False,
             dphase=True,
             mag_only=False):
  """Spectrogram using librosa.
  Args:
    audio: 1-D array of float32 sound samples.
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Mask the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Don't return phase.
  Returns:
    specgram: [n_fft/2 + 1, audio.size / hop_length, 2]. The first channel is
      the logamplitude and the second channel is the derivative of phase.
  """
  if not hop_length:
    hop_length = int(n_fft / 2.)

  fft_config = dict(
      n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

  spec = librosa.stft(audio, **fft_config)

  if re_im:
    re = spec.real[:, :, np.newaxis]
    im = spec.imag[:, :, np.newaxis]
    spec_real = np.concatenate((re, im), axis=2)

  else:
    mag, phase = librosa.core.magphase(spec)
    phase_angle = np.angle(phase)

    # Magnitudes, scaled 0-1
    if log_mag:
      mag = (librosa.power_to_db(
          mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
    else:
      mag /= mag.max()

    if dphase:
      #  Derivative of phase
      phase_unwrapped = np.unwrap(phase_angle)
      p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
      p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
    else:
      # Normal phase
      p = phase_angle / np.pi
    # Mask the phase
    if log_mag and mask:
      p = mag * p
    # Return Mag and Phase
    p = p.astype(np.float32)[:, :, np.newaxis]
    mag = mag.astype(np.float32)[:, :, np.newaxis]
    if mag_only:
      spec_real = mag[:, :, np.newaxis]
    else:
      spec_real = np.concatenate((mag, p), axis=2)
  return spec_real


def inv_magphase(mag, phase_angle):
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  return mag * phase


def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
  """Iterative algorithm for phase retrieval from a magnitude spectrogram.
  Args:
    mag: Magnitude spectrogram.
    phase_angle: Initial condition for phase.
    n_fft: Size of the FFT.
    hop: Stride of FFT. Defaults to n_fft/2.
    num_iters: Griffin-Lim iterations to perform.
  Returns:
    audio: 1-D array of float32 sound samples.
  """
  fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
  ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
  complex_specgram = inv_magphase(mag, phase_angle)
  for i in range(num_iters):
    audio = librosa.istft(complex_specgram, **ifft_config)
    if i != num_iters - 1:
      complex_specgram = librosa.stft(audio, **fft_config)
      _, phase = librosa.magphase(complex_specgram)
      phase_angle = np.angle(phase)
      complex_specgram = inv_magphase(mag, phase_angle)
  return audio


def ispecgram(spec,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=True,
              num_iters=1000):
  """Inverse Spectrogram using librosa.
  Args:
    spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Reverse the mask of the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Specgram contains no phase.
    num_iters: Number of griffin-lim iterations for mag_only.
  Returns:
    audio: 1-D array of sound samples. Peak normalized to 1.
  """
  if not hop_length:
    hop_length = n_fft // 2

  ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

  if mag_only:
    mag = spec[:, :, 0]
    phase_angle = np.pi * np.random.rand(*mag.shape)
  elif re_im:
    spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
  else:
    mag, p = spec[:, :, 0], spec[:, :, 1]
    if mask and log_mag:
      p /= (mag + 1e-13 * np.random.randn(*mag.shape))
    if dphase:
      # Roll up phase
      phase_angle = np.cumsum(p * np.pi, axis=1)
    else:
      phase_angle = p * np.pi

  # Magnitudes
  if log_mag:
    mag = (mag - 1.0) * 120.0
    mag = 10**(mag / 20.0)
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  spec_real = mag * phase

  if mag_only:
    audio = griffin_lim(
        mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
  else:
    audio = librosa.core.istft(spec_real, **ifft_config)
  return np.squeeze(audio / audio.max())


def batch_specgram(audio,
                   n_fft=512,
                   hop_length=None,
                   mask=True,
                   log_mag=True,
                   re_im=False,
                   dphase=True,
                   mag_only=False):
  """Computes specgram in a batch."""
  assert len(audio.shape) == 2
  batch_size = audio.shape[0]
  res = []
  for b in range(batch_size):
    res.append(
        specgram(audio[b], n_fft, hop_length, mask, log_mag, re_im, dphase,
                 mag_only))
  return np.array(res)


def batch_ispecgram(spec,
                    n_fft=512,
                    hop_length=None,
                    mask=True,
                    log_mag=True,
                    re_im=False,
                    dphase=True,
                    mag_only=False,
                    num_iters=1000):
  """Computes inverse specgram in a batch."""
  assert len(spec.shape) == 4
  batch_size = spec.shape[0]
  res = []
  for b in range(batch_size):
    res.append(
        ispecgram(spec[b, :, :, :], n_fft, hop_length, mask, log_mag, re_im,
                  dphase, mag_only, num_iters))
  return np.array(res)


def tf_specgram(audio,
                n_fft=512,
                hop_length=None,
                mask=True,
                log_mag=True,
                re_im=False,
                dphase=True,
                mag_only=False):
  """Specgram tensorflow op (uses pyfunc)."""
  return batch_specgram(
      audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only)


def tf_ispecgram(spec,
                 n_fft=512,
                 hop_length=None,
                 mask=True,
                 pad=True,
                 log_mag=True,
                 re_im=False,
                 dphase=True,
                 mag_only=False,
                 num_iters=1000):
  """Inverted Specgram tensorflowtensorflow op (uses pyfunc)."""
  dims = spec.get_shape().as_list()
  # Add back in nyquist frequency
  if pad:
    x = tf.concat([spec, tf.zeros([dims[0], 1, dims[2], dims[3]])], 1)
  else:
    x = spec
  audio = tf.py_func(batch_ispecgram, [
      x, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only, num_iters
  ], tf.float32)
  return audio