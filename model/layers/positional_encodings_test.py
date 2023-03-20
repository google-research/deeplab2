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

"""Tests for positional_encodings."""

import tensorflow as tf

from deeplab2.model.layers import positional_encodings


class PositionalEncodingsTest(tf.test.TestCase):

  def test_compute_relative_distance_matrix_output_shape(self):
    output = positional_encodings._compute_relative_distance_matrix(33, 97)
    self.assertListEqual(output.get_shape().as_list(), [33, 97])

  def test_relative_positional_encoding_output_shape(self):
    layer = positional_encodings.RelativePositionalEncoding(
        33, 97, 32, 'rpe')
    output = layer(None)
    self.assertListEqual(output.get_shape().as_list(), [33, 97, 32])

  def test_add_absolute_positional_encoding_1d_output_shape(self):
    layer = positional_encodings.AddAbsolutePositionalEncoding(
        'ape1d', positional_encoding_type='1d')
    shape = [2, 5, 5, 3]
    output = layer(tf.zeros(shape))
    self.assertEqual(len(layer.get_weights()), 10)
    self.assertListEqual(output.get_shape().as_list(), shape)

  def test_add_absolute_positional_encoding_2d_output_shape(self):
    layer = positional_encodings.AddAbsolutePositionalEncoding(
        'ape2d', positional_encoding_type='2d')
    shape = [2, 5, 5, 3]
    output = layer(tf.zeros(shape))
    self.assertEqual(len(layer.get_weights()), 5)
    self.assertListEqual(output.get_shape().as_list(), shape)

  def test_add_absolute_positional_encoding_none_output_shape(self):
    layer = positional_encodings.AddAbsolutePositionalEncoding(
        'none', positional_encoding_type='none')
    shape = [2, 5, 5, 3]
    output = layer(tf.zeros(shape))
    self.assertEqual(len(layer.get_weights()), 0)
    self.assertListEqual(output.get_shape().as_list(), shape)

if __name__ == '__main__':
  tf.test.main()
