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

"""Tests for moat_blocks."""

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from deeplab2.model.layers import moat_blocks


class MOATBlocksTest(tf.test.TestCase, parameterized.TestCase):

  def _log_param_specs(self, layer):
    num_params = sum([
        np.prod(var.get_shape().as_list()) for var in layer.trainable_weights
    ])
    format_str = '{{:<{0}s}}\t{{:<{1}s}}'.format(
        max([len(v.name) for v in layer.trainable_weights]),
        max([len('{}'.format(v.get_shape())) for v in
             layer.trainable_weights]))
    format_str = '  >> ' + format_str + '\t{:>5.2f}%'

    for v in layer.trainable_weights:
      v_shape = v.get_shape().as_list()
      logging.info(format_str.format(v.name, '{}'.format(v_shape),
                                     np.prod(v_shape) / num_params * 100))

  @parameterized.named_parameters(
      ('standard', 1),
      ('downsample', 2),
  )
  def test_mbconv_block(self, stride):
    batch_size = 8
    height, width = 8, 8
    input_size = 16
    hidden_size = input_size * stride
    inputs = tf.random.normal(shape=[batch_size, height, width, input_size],
                              dtype=tf.float32)
    block = moat_blocks.MBConvBlock(hidden_size=hidden_size,
                                    block_stride=stride,)
    output = block(inputs, training=True)
    self._log_param_specs(block)

    self.assertEqual(output.shape.as_list(),
                     [batch_size, height // stride, width // stride,
                      hidden_size])

  @parameterized.named_parameters(
      ('standard', 1, False),
      ('downsample', 2, False),
      ('checkpointing', 1, True),
  )
  def test_moat_block(self, stride, use_checkpointing):
    batch_size = 8
    height, width = 8, 8
    input_size = 16
    hidden_size = input_size * stride
    inputs = tf.random.normal(shape=[batch_size, height, width, input_size],
                              dtype=tf.float32)
    block = moat_blocks.MOATBlock(hidden_size=hidden_size,
                                  block_stride=stride,
                                  window_size=[height//stride, width//stride],
                                  use_checkpointing=use_checkpointing)
    output = block(inputs, training=True)
    self._log_param_specs(block)

    self.assertEqual(output.shape.as_list(),
                     [batch_size, height // stride, width // stride,
                      hidden_size])


if __name__ == '__main__':
  tf.test.main()
