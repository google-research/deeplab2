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

"""Tests for squeeze_and_excite.py."""

import tensorflow as tf

from deeplab2.model.layers import squeeze_and_excite


class SqueezeAndExciteTest(tf.test.TestCase):

  def test_simpliefied_squeeze_and_excite_input_output_shape(self):
    # Test the shape of input and output of SimplifiedSqueezeAndExcite.
    channels = 32
    input_tensor = tf.random.uniform(shape=(3, 65, 65, channels))
    layer_op = squeeze_and_excite.SimplifiedSqueezeAndExcite(
        channels)
    output_tensor = layer_op(input_tensor)
    self.assertListEqual(input_tensor.get_shape().as_list(),
                         output_tensor.get_shape().as_list())

  def test_squeeze_and_excite_input_output_shape(self):
    # Test the shape of input and output of SqueezeAndExcite.
    channels = 32
    input_tensor = tf.random.uniform(shape=(3, 65, 65, channels))
    layer_op = squeeze_and_excite.SqueezeAndExcite(
        in_filters=channels,
        out_filters=channels,
        se_ratio=8,
        name='se')
    output_tensor = layer_op(input_tensor)
    self.assertListEqual(input_tensor.get_shape().as_list(),
                         output_tensor.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
