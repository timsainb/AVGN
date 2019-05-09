import tensorflow as tf
import tensorflow_datasets as tfds

if int(tf.__version__[0]) < 2:
    from tensorflow import FixedLenFeature, parse_single_example
else:
    from tensorflow.io import FixedLenFeature, parse_single_example

from avgn.signalprocessing.spectrogramming_tf import spectrogram_tensorflow


class HParams(object):
    """ Hparams was removed from tf 2.0 so this is a placeholder
	"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class NSynthDataset(object):
    def __init__(self, tf_records, hparams, is_training=True, prefetch = 1000, num_parallel_calls = 10):
        self.is_training = is_training
        self.hparams = hparams
        # prepare for mel scaling
        if self.hparams.mel:
        	self.mel_matrix = self._make_mel_matrix()
        # create dataset of tfrecords
        self.raw_dataset = tf.data.TFRecordDataset(tf_records)
        # prepare dataset iterations
        self.dataset = self.raw_dataset.map(lambda x: self._parse_function(x), num_parallel_calls=num_parallel_calls)
        self.dataset_batch = self.dataset.shuffle(10000)
        self.dataset_batch = self.dataset_batch.prefetch(prefetch)
        self.dataset_batch = self.dataset_batch.batch(hparams.batch_size)
        

    def _make_mel_matrix(self):
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.hparams.mel_matrix_dict["num_mel_bins"],
            num_spectrogram_bins=self.hparams.mel_matrix_dict["num_spectrogram_bins"],
            sample_rate=self.hparams.mel_matrix_dict["sample_rate"],
            lower_edge_hertz=self.hparams.mel_matrix_dict["lower_edge_hertz"],
            upper_edge_hertz=self.hparams.mel_matrix_dict["upper_edge_hertz"],
            dtype=tf.dtypes.float32,
            name=None,
        )
        return mel_matrix

    def print_feature_list(self):
        # get the features
        element = list(self.raw_dataset.take(count=1))[0]
        # parse the element in to the example message
        example = tf.train.Example()
        example.ParseFromString(element.numpy())
        print(list(example.features.feature))

    def _parse_function(self, example_proto):
        features = {
            "id": FixedLenFeature([], dtype=tf.string),
            "pitch": FixedLenFeature([1], dtype=tf.int64),
            "velocity": FixedLenFeature([1], dtype=tf.int64),
            "audio": FixedLenFeature([64000], dtype=tf.float32),
            # "qualities": FixedLenFeature([10], dtype=tf.int64),
            "instrument/source": FixedLenFeature([1], dtype=tf.int64),
            "instrument/family": FixedLenFeature([1], dtype=tf.int64),
            "instrument/label": FixedLenFeature([1], dtype=tf.int64),
        }
        example = parse_single_example(example_proto, features)

        if self.hparams.spectrogram:
            example["spectrogram"] = spectrogram_tensorflow(
                example["audio"], self.hparams
            )
            if self.hparams.mel:
                example["spectrogram"] = tf.tensordot(
                    example["spectrogram"], self.mel_matrix, 1
                )
                if self.hparams.mfcc:
                    example["spectrogram"] = tf.signal.mfccs_from_log_mel_spectrograms(
                        example["spectrogram"]
                    )
            example["spectrogram"] = example["spectrogram"] / tf.reduce_max(example["spectrogram"])
        return example


def download_nsynth(save_loc):
    ### TO INITIALLY DOWNLOAD THE DATASET
    # Construct a tf.data.Dataset
    ds_train, ds_test = tfds.load(
        name="nsynth", split=["train", "test"], data_dir=save_loc
    )
