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

"""Tests for axial_layers."""

import tensorflow as tf

from deeplab2.model.layers import axial_layers


class AxialLayersTest(tf.test.TestCase):

  def test_default_axial_attention_layer_output_shape(self):
    layer = axial_layers.AxialAttention()
    output = layer(tf.zeros([10, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [10, 5, 1024])

  def test_axial_attention_2d_layer_output_shape(self):
    layer = axial_layers.AxialAttention2D()
    output = layer(tf.zeros([2, 5, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [2, 5, 5, 1024])

  def test_change_filters_output_shape(self):
    layer = axial_layers.AxialAttention2D(filters=32)
    output = layer(tf.zeros([2, 5, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [2, 5, 5, 64])

  def test_value_expansion_output_shape(self):
    layer = axial_layers.AxialAttention2D(value_expansion=1)
    output = layer(tf.zeros([2, 5, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [2, 5, 5, 512])

  def test_global_attention_output_shape(self):
    layer = axial_layers.GlobalAttention2D()
    output = layer(tf.zeros([2, 5, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [2, 5, 5, 1024])

  def test_stride_two_output_shape(self):
    layer = axial_layers.AxialAttention2D(strides=2)
    output = layer(tf.zeros([2, 5, 5, 32]))
    self.assertListEqual(output.get_shape().as_list(), [2, 3, 3, 1024])

if __name__ == '__main__':
  tf.test.main()
