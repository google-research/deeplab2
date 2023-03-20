# coding=utf-8
# Copyright 2023 The Deeplab2 Authors.
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

"""Input reader to load segmentation dataset with multicamera format."""

import tensorflow as tf

_NUM_INPUTS_PROCESSED_CONCURRENTLY = 32
_SHUFFLE_BUFFER_SIZE = 1000


class MultiCameraInputReader(object):
  """Creates a dataset from files with multiple cameras' info."""

  def __init__(self,
               file_pattern,
               decoder_fn,
               generator_fn=None,
               camera_names=None,
               use_panoptic_copy_paste=False,
               is_training=False):
    """Initializes the input reader.

    Args:
      file_pattern: The file pattern for the data example, in TFRecord format
      decoder_fn: A callable that takes a serialized tf.Example and produces
        parsed (and potentially processed / augmented) tensors.
      generator_fn: An optional `callable` that takes the decoded raw tensors
        dict and generates a ground-truth dictionary that can be consumed by the
        model. It will be executed after decoder_fn (default: None).
      camera_names: A list of camera name strings to keep in the dataset. If
        None, we will not flatten the dataset but keep the `camera_name:
        camera_value` structure, where each example contains all camera
        information. Otherwise, we will regard each view, specified by the
        camera_names, as individual examples.
      use_panoptic_copy_paste: If the panoptic_copy_paste augmentation is used
        or not (default: False).
      is_training: If this dataset is used for training or not (default: False).
    """
    self._file_pattern = file_pattern
    self._is_training = is_training
    self._decoder_fn = decoder_fn
    self._generator_fn = generator_fn
    self._camera_names = camera_names
    self._use_panoptic_copy_paste = use_panoptic_copy_paste
    if self._use_panoptic_copy_paste:
      raise NotImplementedError(
          'use_panoptic_copy_paste is not supported for the multicamera format.'
      )

  def __call__(self, batch_size=1, max_num_examples=-1):
    """Provides tf.data.Dataset object.

    Args:
      batch_size: Expected batch size input data.
      max_num_examples: Positive integer or -1. If positive, the returned
        dataset will only take (at most) this number of examples and raise
        tf.errors.OutOfRangeError after that (default: -1).

    Returns:
      tf.data.Dataset object.
    """
    def decode_dataset():
      dataset = tf.data.Dataset.list_files(
          self._file_pattern, shuffle=self._is_training)

      if self._is_training:
        # File level shuffle.
        dataset = dataset.shuffle(dataset.cardinality(),
                                  reshuffle_each_iteration=True)
        dataset = dataset.repeat()

      # During training, interleave TFRecord conversion for maximum efficiency.
      # During evaluation, read input in consecutive order for tasks requiring
      # such behavior.
      dataset = dataset.interleave(
          map_func=tf.data.TFRecordDataset,
          cycle_length=(_NUM_INPUTS_PROCESSED_CONCURRENTLY
                        if self._is_training else 1),
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=not self._is_training)

      if self._is_training:
        dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
      if max_num_examples > 0:
        dataset = dataset.take(max_num_examples)

      # Parses the fetched records to input tensors for model function.
      dataset = dataset.map(
          self._decoder_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    dataset = decode_dataset()
    if self._generator_fn is not None:
      # If panoptic copy-paste is enabled, we duplicate the dataset with another
      # random order, and zip them together. Thus, the generator will get a
      # tuple containing two samples, which can be used for panoptic copy-paste
      # augmentations.
      if self._use_panoptic_copy_paste:
        panoptic_copy_paste_dataset = decode_dataset()
        dataset = tf.data.Dataset.zip((dataset, panoptic_copy_paste_dataset))
      dataset = dataset.map(
          self._generator_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._camera_names:

      def _select_camera_data(dataset, camera_name):
        return dataset.map(
            lambda val: val[camera_name],
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Flat and keep only cameras we need.
      view_datasets = [
          _select_camera_data(dataset, camera_name)
          for camera_name in self._camera_names
      ]
      dataset = view_datasets[0]
      for other_dataset in view_datasets[1:]:
        dataset = dataset.concatenate(other_dataset)
      if self._is_training:
        dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE *
                                  len(self._camera_names))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
