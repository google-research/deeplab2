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

"""Tests for moat."""

import re
from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import
from deeplab2.model.pixel_encoder import moat as moat_lib

tf.compat.v1.disable_v2_behavior()
FLAGS = flags.FLAGS


@ops.RegisterStatistics('Einsum', 'flops')
def _einsum_flops(graph, node):
  """Computes flops for einsum ops."""

  def _get_shape(name):
    shape = tf.compat.v1.graph_util.tensor_shape_from_node_def_name(graph, name)
    shape.assert_is_fully_defined()
    return shape.as_list()

  if len(node.input) != 2:
    raise ValueError('The lengh of input should be 2')
  u_shape = _get_shape(node.input[0])
  v_shape = _get_shape(node.input[1])
  # ab,bc->ac
  equation = node.attr['equation'].s.decode('utf-8')
  # ab, bc, ac
  u_expr, v_expr, o_expr = re.split(',|->', equation)
  # {a, b, c}
  inputs_chrs = set(u_expr + v_expr)
  # {a, c}
  output_chrs = set(o_expr)
  # {b}
  contracted_chrs = inputs_chrs - output_chrs
  chr2dim = {}
  for c in inputs_chrs:
    if c in u_expr:
      dim = u_shape[u_expr.index(c)]
    else:
      dim = v_shape[v_expr.index(c)]
    chr2dim[c] = dim
  output_dims = [chr2dim[c] for c in output_chrs]
  contracted_dims = [chr2dim[c] for c in contracted_chrs]
  flops = 2 * np.prod(output_dims) * np.prod(contracted_dims)  # 2 * madds
  return ops.OpStats('flops', flops)


class MOATTest(tf.test.TestCase, parameterized.TestCase):

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

  def _set_precision_policy(self, policy_name=None):
    """Sets precision policy according to the name."""

    if policy_name not in ['mixed_float16', 'mixed_bfloat16', 'float32']:
      raise ValueError('Unsupported policy name: %s' % policy_name)
    logging.info('use mixed precision policy name %s', policy_name)
    tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
    tf.keras.mixed_precision.set_global_policy(policy_name)

  def _profile_default_graph(self, batch_size):
    """Computes flops and params of the default graph."""

    profile = {}
    profile['params'] = np.sum(
        [np.prod(var.get_shape().as_list()) for var in
         tf.compat.v1.trainable_variables()])
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    flops = tf.compat.v1.profiler.profile(
        tf.compat.v1.get_default_graph(), options=options).total_float_ops
    profile['flops'] = flops / batch_size / 2
    return profile

  @parameterized.parameters(
      ('moat0_pretrain_224_1k', '224x224', 27_846_892),
      ('moat1_pretrain_224_1k', '224x224', 41_598_448),
      ('moat2_finetune_384_22k', '384x384', 73_422_408),
      ('moat3_finetune_512_22k', '512x512', 189_958_040),
      ('moat4_finetune_512_22k', '512x512', 483_214_184),
      ('moat0_pretrain_224_no_pe_1k', '224x224', 27_777_544),
      ('moat1_pretrain_224_no_pe_1k', '224x224', 41_467_864),
      ('moat2_finetune_384_no_pe_22k', '384x384', 73_248_296),
      ('moat3_finetune_512_no_pe_22k', '512x512', 189_536_280),
      ('moat4_finetune_512_no_pe_22k', '512x512', 482_539_368),
      ('tiny_moat0_pretrain_256_1k', '256x256', 3_355_348),
      ('tiny_moat1_pretrain_256_1k', '256x256', 5_122_895),
      ('tiny_moat2_pretrain_256_1k', '256x256', 9_772_741),
      ('tiny_moat3_pretrain_256_1k', '256x256', 19_534_390),
      ('tiny_moat0_pretrain_256_no_pe_1k', '256x256', 3_332_232),
      ('tiny_moat1_pretrain_256_no_pe_1k', '256x256', 5_094_000),
      ('tiny_moat2_pretrain_256_no_pe_1k', '256x256', 9_732_288),
      ('tiny_moat3_pretrain_256_no_pe_1k', '256x256', 19_476_600),
  )
  def test_moat_with_output_shape_and_params(
      self, name, resolution, expected_params):
    batch_size = 8
    height, width = map(int, resolution.split('x'))
    input_size = 3
    num_classes = 1_000
    override_config = dict(
        build_classification_head_with_class_num=num_classes)

    inputs = tf.random.normal(shape=[batch_size, height, width, input_size],
                              dtype=tf.float32)

    # We test the model with mixed_bfloat16 precision.
    self._set_precision_policy('mixed_bfloat16')
    inputs = tf.cast(inputs, tf.bfloat16)
    with tf.compat.v1.tpu.bfloat16_scope():
      moat = moat_lib.get_model(
          name,
          input_shape=(height, width, input_size),
          window_size=[None, None,
                       [height//16, width//16],
                       [height//32, width//32]],
          override_config=override_config,
          pretrained_weights_path=None,
          global_attention_at_end_of_moat_stage=True,
          use_checkpointing_for_attention=True,
      )
      output = moat(inputs, training=False)
    output = tf.cast(output, tf.float32)
    self._set_precision_policy('float32')

    profile = self._profile_default_graph(batch_size=batch_size)
    logging.info('=' * 120)
    logging.info('#FLOPs: {:,}, #Params: {:,}'.format(
        profile['flops'], profile['params']))
    logging.info('=' * 120)

    self._log_param_specs(moat)
    self.assertEqual(output.shape.as_list(), [batch_size, num_classes])
    self.assertEqual(profile['params'], expected_params)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
