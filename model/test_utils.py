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

"""This file contains utility functions for the model tests."""
import numpy as np
import tensorflow as tf


def create_test_input(batch, height, width, channels):
  """Creates test input tensor."""
  input_tensor = np.tile(
      np.reshape(
          np.reshape(np.arange(height), [height, 1]) +
          np.reshape(np.arange(width), [1, width]),
          [1, height, width, 1]),
      [batch, 1, 1, channels])
  # Normalize the input tensor so that the outputs are not too large.
  input_tensor = (input_tensor * 2 / np.max(input_tensor)) - 1
  return tf.cast(input_tensor, tf.float32)
