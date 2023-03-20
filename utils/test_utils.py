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

"""Provide utility functions to write simple tests."""
import functools

import numpy as np
import tensorflow as tf


NORMALIZATION_LAYERS = (
    tf.keras.layers.experimental.SyncBatchNormalization,
    tf.keras.layers.BatchNormalization
)


def create_strategy():
  """Returns a strategy based on available devices.

  Does NOT work with local_multiworker_tpu_test tests!
  """
  tpus = tf.config.list_logical_devices(device_type='TPU')
  gpus = tf.config.list_logical_devices(device_type='GPU')
  if tpus:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)
  elif gpus:
    return tf.distribute.OneDeviceStrategy('/gpu:0')
  else:
    return tf.distribute.OneDeviceStrategy('/cpu:0')


def test_all_strategies(func):
  """Decorator to test CPU, GPU and TPU strategies."""
  @functools.wraps(func)
  def decorator(self):
    strategy = create_strategy()
    return func(self, strategy)
  return decorator


def create_test_input(batch, height, width, channels):
  """Creates test input tensor."""
  return tf.convert_to_tensor(
      np.tile(
          np.reshape(
              np.reshape(np.arange(height), [height, 1]) +
              np.reshape(np.arange(width), [1, width]),
              [1, height, width, 1]),
          [batch, 1, 1, channels]), dtype=tf.float32)
