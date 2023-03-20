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

"""Tests for axial_block_groups."""

import numpy as np
import tensorflow as tf

from deeplab2.model import test_utils
from deeplab2.model.layers import axial_block_groups


class AxialBlockGroupsTest(tf.test.TestCase):

  def test_axial_attention_follows_bottleneck_block(self):
    layer = axial_block_groups.BlockGroup(
        filters=512,
        num_blocks=2,
        name='block_group',
        original_resnet_stride=2,
        original_resnet_input_stride=16,
        use_axial_beyond_stride=32,
        output_stride=16)
    pixel_output, memory_output = layer(
        (tf.zeros([2, 65, 65, 1024]), tf.zeros([2, 128, 147])))
    self.assertListEqual(pixel_output.get_shape().as_list(), [2, 65, 65, 2048])
    self.assertListEqual(memory_output.get_shape().as_list(), [2, 128, 147])

  def test_global_attention_follows_basic_block(self):
    layer = axial_block_groups.BlockGroup(
        filters=256,
        num_blocks=2,
        name='block_group',
        backbone_type='wider_resnet',
        original_resnet_stride=2,
        original_resnet_input_stride=8,
        use_global_beyond_stride=16,
        positional_encoding_type='1D')

    pixel_output, memory_output = layer(
        (tf.zeros([2, 65, 65, 32]), tf.zeros([2, 128, 147])))
    self.assertListEqual(pixel_output.get_shape().as_list(), [2, 33, 33, 1024])
    self.assertListEqual(memory_output.get_shape().as_list(), [2, 128, 147])

  def test_atrous_consistency_basic_block(self):
    tf.random.set_seed(0)
    pixel_inputs = test_utils.create_test_input(2, 11, 11, 3)
    # Dense feature extraction followed by subsampling.
    layer1 = axial_block_groups.BlockGroup(
        filters=2,
        num_blocks=2,
        name='stage3',
        backbone_type='wider_resnet',
        original_resnet_stride=2,
        original_resnet_input_stride=8,
        output_stride=8,
        use_axial_beyond_stride=0,
        use_global_beyond_stride=0,
        use_transformer_beyond_stride=0)
    # Create the weights
    layer1((pixel_inputs, None))
    weights = layer1.get_weights()
    # Set the batch norm gamma as non-zero so that the 3x3 convolution affects
    # the output.
    for index in range(len(weights)):
      if np.sum(weights[index]) == 0.0:
        weights[index] = weights[index] + 1
    layer1.set_weights(weights)
    pixel_outputs, _ = layer1((pixel_inputs, None))
    output = pixel_outputs[:, ::2, ::2, :]
    # Feature extraction at the nominal network rate.
    layer2 = axial_block_groups.BlockGroup(
        filters=2,
        num_blocks=2,
        name='stage3',
        backbone_type='wider_resnet',
        original_resnet_stride=2,
        original_resnet_input_stride=8,
        output_stride=16,
        use_axial_beyond_stride=0,
        use_global_beyond_stride=0,
        use_transformer_beyond_stride=0)
    # Create the weights
    layer2((pixel_inputs, None))
    # Make the two networks use the same weights.
    layer2.set_weights(layer1.get_weights())
    expected, _ = layer2((pixel_inputs, None))
    self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

  def test_atrous_consistency_bottleneck_block(self):
    tf.random.set_seed(0)
    pixel_inputs = test_utils.create_test_input(2, 11, 11, 3)
    # Dense feature extraction followed by subsampling.
    layer1 = axial_block_groups.BlockGroup(
        filters=2,
        num_blocks=2,
        name='stage3',
        backbone_type='wider_resnet',
        original_resnet_stride=2,
        original_resnet_input_stride=16,
        output_stride=16,
        use_axial_beyond_stride=0,
        use_global_beyond_stride=0,
        use_transformer_beyond_stride=0)
    # Create the weights
    layer1((pixel_inputs, None))
    weights = layer1.get_weights()
    # Set the batch norm gamma as non-zero so that the 3x3 convolution affects
    # the output.
    for index in range(len(weights)):
      if np.sum(weights[index]) == 0.0:
        weights[index] = weights[index] + 1
    layer1.set_weights(weights)
    pixel_outputs, _ = layer1((pixel_inputs, None))
    output = pixel_outputs[:, ::2, ::2, :]
    # Feature extraction at the nominal network rate.
    layer2 = axial_block_groups.BlockGroup(
        filters=2,
        num_blocks=2,
        name='stage3',
        backbone_type='wider_resnet',
        original_resnet_stride=2,
        original_resnet_input_stride=16,
        output_stride=32,
        use_axial_beyond_stride=0,
        use_global_beyond_stride=0,
        use_transformer_beyond_stride=0)
    # Create the weights
    layer2((pixel_inputs, None))
    # Make the two networks use the same weights.
    layer2.set_weights(layer1.get_weights())
    expected, _ = layer2((pixel_inputs, None))
    self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

  def test_use_se_sac_recompute_drop_path_schedule(self):
    _ = axial_block_groups.BlockGroup(
        filters=512,
        num_blocks=2,
        name='block_group',
        original_resnet_stride=2,
        original_resnet_input_stride=8,
        use_axial_beyond_stride=0,
        use_squeeze_and_excite=True,  # True
        use_sac_beyond_stride=16,  # True
        recompute_within_stride=16,  # True
        drop_path_beyond_stride=16,
        drop_path_schedule='linear',  # 1.0, 0.85
        output_stride=16)

  def test_nouse_se_sac_recompute_drop_path_schedule(self):
    _ = axial_block_groups.BlockGroup(
        filters=512,
        num_blocks=2,
        name='block_group',
        original_resnet_stride=2,
        original_resnet_input_stride=8,
        use_axial_beyond_stride=0,
        use_squeeze_and_excite=False,  # False
        use_sac_beyond_stride=32,  # False
        recompute_within_stride=8,  # False
        drop_path_beyond_stride=32,  # 1.0, 1.0
        drop_path_schedule='constant',
        output_stride=16)

if __name__ == '__main__':
  tf.test.main()
