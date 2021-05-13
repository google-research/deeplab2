# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
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

"""Tests for axial_blocks."""

import tensorflow as tf

from deeplab2.model.layers import axial_blocks


class AxialBlocksTest(tf.test.TestCase):

  def test_conv_basic_block_correct_output_shape(self):
    layer = axial_blocks.AxialBlock(
        filters_list=[256, 256],
        strides=2,
        drop_path_keep_prob=0.8,
        recompute_grad=True)
    output = layer(tf.zeros([2, 65, 65, 32]))[1]
    self.assertListEqual(output.get_shape().as_list(), [2, 33, 33, 256])

  def test_conv_bottleneck_block_correct_output_shape(self):
    layer = axial_blocks.AxialBlock(
        filters_list=[64, 64, 256],
        strides=1,
        drop_path_keep_prob=1.0,
        recompute_grad=False)
    output = layer(tf.zeros([2, 65, 65, 32]))[0]
    self.assertListEqual(output.get_shape().as_list(), [2, 65, 65, 256])

  def test_axial_block_correct_output_shape(self):
    layer = axial_blocks.AxialBlock(
        filters_list=[128, 64, 256],
        strides=2,
        drop_path_keep_prob=1.0,
        recompute_grad=False,
        attention_type='axial')
    output = layer(tf.zeros([2, 65, 65, 32]))[1]
    self.assertListEqual(output.get_shape().as_list(), [2, 33, 33, 256])

if __name__ == '__main__':
  tf.test.main()
