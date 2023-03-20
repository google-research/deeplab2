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

"""Tests for moat_attention."""

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from deeplab2.model.layers import moat_attention


class MOATAttentionTest(tf.test.TestCase, parameterized.TestCase):

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
      ('attention', None),
      ('attention_with_relative_position_embedding', '2d_multi_head'),
  )
  def test_attention(self, relative_position_embedding_type):
    batch_size = 8
    height = 8
    width = 10
    hidden_size = 16
    head_size = 8
    query = tf.random.normal(shape=[batch_size, height, width, hidden_size],
                             dtype=tf.float32)

    attention_layer = moat_attention.Attention(
        hidden_size=hidden_size,
        head_size=head_size,
        relative_position_embedding_type=relative_position_embedding_type)
    attention_output = attention_layer(query, training=True)
    self._log_param_specs(attention_layer)

    self.assertEqual(attention_output.shape.as_list(),
                     [batch_size, height * width, hidden_size])

if __name__ == '__main__':
  tf.test.main()
