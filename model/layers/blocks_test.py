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

"""Tests for blocks.py."""
import tensorflow as tf

from deeplab2.model.layers import blocks


class BlocksTest(tf.test.TestCase):

  def test_inverted_bottleneck_block_output_shape(self):
    batch, height, width, input_channels = 2, 17, 17, 4
    output_channels = 6
    input_tensor = tf.random.uniform(
        shape=(batch, height, width, input_channels))
    ivb_block = blocks.InvertedBottleneckBlock(
        in_filters=input_channels,
        out_filters=output_channels,
        expand_ratio=2,
        strides=1,
        name='inverted_bottleneck',
    )
    output_tensor, _ = ivb_block(input_tensor)
    self.assertListEqual(output_tensor.get_shape().as_list(),
                         [batch, height, width, output_channels])

  def test_inverted_bottleneck_block_feature_map_alignment(self):
    batch, height, width, input_channels = 2, 17, 17, 128
    output_channels = 256
    input_tensor = tf.random.uniform(
        shape=(batch, height, width, input_channels))
    ivb_block1 = blocks.InvertedBottleneckBlock(
        in_filters=input_channels,
        out_filters=output_channels,
        expand_ratio=2,
        strides=2,
        name='inverted_bottleneck1',
    )
    ivb_block1(input_tensor, False)
    weights = ivb_block1.get_weights()
    output_tensor, _ = ivb_block1(input_tensor, False)

    ivb_block2 = blocks.InvertedBottleneckBlock(
        in_filters=input_channels,
        out_filters=output_channels,
        expand_ratio=2,
        strides=1,
        name='inverted_bottleneck2',
    )
    ivb_block2(input_tensor, False)
    ivb_block2.set_weights(weights)
    expected = ivb_block2(input_tensor, False)[0][:, ::2, ::2, :]

    self.assertAllClose(ivb_block1.get_weights(), ivb_block2.get_weights(),
                        atol=1e-4, rtol=1e-4)
    self.assertAllClose(output_tensor, expected, atol=1e-4, rtol=1e-4)

if __name__ == '__main__':
  tf.test.main()
