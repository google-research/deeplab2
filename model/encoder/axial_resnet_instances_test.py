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

"""Tests for axial_resnet_instances."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deeplab2.model.encoder import axial_resnet_instances


class AxialResnetInstancesTest(tf.test.TestCase, parameterized.TestCase):

  # The parameter count does not include the classification head.
  @parameterized.parameters(
      ('resnet50', 1, 23508032),
      ('resnet50_beta', 1, 23631808),  # 123776 more than resnet50
      ('max_deeplab_s_backbone', 1, 41343424),
      ('max_deeplab_l_backbone', 1, 175115392),
      ('axial_resnet_s', 1, 11466912),
      ('axial_resnet_l', 1, 43714048),  # 127872 fewer than axial_deeplab_l
      ('axial_deeplab_s', 1, 11565856),
      ('axial_deeplab_l', 1, 43841920),
      ('swidernet', 1, 109014080),  # SWideRNet-(1,1,1) without SE or SAC
      ('swidernet', 3, 333245504),  # Should be more than 3 x 109014080
      ('swidernet', 4.5, 487453760),  # Rounded down to [13, 27, 13, 13]
      ('axial_swidernet', 1, 136399392),
      ('axial_swidernet', 3, 393935520),
      ('axial_swidernet', 4.5, 570346912),
      )
  def test_model_output_shape_and_num_params(
      self, model_name, backbone_layer_multiplier, expected_num_params):
    model = axial_resnet_instances.get_model(
        model_name,
        backbone_layer_multiplier=backbone_layer_multiplier,
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001)
    output = model(tf.keras.Input(shape=(224, 224, 3)))
    if model_name in ('axial_resnet_s', 'axial_deeplab_s'):
      self.assertListEqual(output['res5'].get_shape().as_list(),
                           [None, 14, 14, 1024])
    else:
      self.assertListEqual(output['res5'].get_shape().as_list(),
                           [None, 14, 14, 2048])
    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, expected_num_params)

  def test_resnet50_variable_checkpoint_names(self):
    model = axial_resnet_instances.get_model(
        'resnet50',
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001)
    model(tf.keras.Input(shape=(224, 224, 3)))
    variable_names = [w.name for w in model.trainable_weights]
    test_variable_name = 'resnet50/stage4/block6/conv3_bn/batch_norm/beta:0'
    self.assertIn(test_variable_name, variable_names)
    temp_dir = self.create_tempdir()
    temp_path = os.path.join(temp_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(encoder=model)
    checkpoint.save(temp_path)
    latest_checkpoint = tf.train.latest_checkpoint(temp_dir)
    reader = tf.train.load_checkpoint(latest_checkpoint)
    checkpoint_names = reader.get_variable_to_shape_map().keys()
    test_checkpoint_name = 'encoder/_stage4/_block6/_conv3_bn/_batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
    self.assertIn(test_checkpoint_name, checkpoint_names)

  def test_max_deeplab_s_output_shape_and_num_params(self):
    model = axial_resnet_instances.get_model(
        'max_deeplab_s',
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001)
    endpoints = model(tf.keras.Input(shape=(65, 65, 3)))
    self.assertListEqual(endpoints['backbone_output'].get_shape().as_list(),
                         [None, 5, 5, 2048])
    self.assertListEqual(
        endpoints['transformer_class_feature'].get_shape().as_list(),
        [None, 128, 256])
    self.assertListEqual(
        endpoints['transformer_mask_feature'].get_shape().as_list(),
        [None, 128, 256])
    self.assertListEqual(endpoints['feature_panoptic'].get_shape().as_list(),
                         [None, 17, 17, 256])
    self.assertListEqual(endpoints['feature_semantic'].get_shape().as_list(),
                         [None, 5, 5, 2048])
    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, 61726624)

  def test_max_deeplab_l_output_shape_and_num_params(self):
    model = axial_resnet_instances.get_model(
        'max_deeplab_l',
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001)
    endpoints = model(tf.keras.Input(shape=(65, 65, 3)))
    self.assertListEqual(endpoints['backbone_output'].get_shape().as_list(),
                         [None, 5, 5, 2048])
    self.assertListEqual(
        endpoints['transformer_class_feature'].get_shape().as_list(),
        [None, 128, 512])
    self.assertListEqual(
        endpoints['transformer_mask_feature'].get_shape().as_list(),
        [None, 128, 512])
    self.assertListEqual(endpoints['feature_panoptic'].get_shape().as_list(),
                         [None, 17, 17, 256])
    self.assertListEqual(endpoints['feature_semantic'].get_shape().as_list(),
                         [None, 17, 17, 256])
    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, 450523232)

  def test_global_attention_absolute_positional_encoding_names(self):
    model = axial_resnet_instances.get_model(
        'max_deeplab_s_backbone',
        block_group_config={'use_global_beyond_stride': 16,
                            'positional_encoding_type': '1D',
                            'axial_layer_config': {
                                'use_query_rpe_similarity': False,
                                'use_key_rpe_similarity': False,
                                'retrieve_value_rpe': False}},
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001)
    model(tf.keras.Input(shape=(224, 224, 3)))
    variable_names = [w.name for w in model.trainable_weights]
    test_variable_name1 = 'max_deeplab_s_backbone/stage4/add_absolute_positional_encoding/height_axis_embeddings:0'
    test_variable_name2 = 'max_deeplab_s_backbone/stage4/block2/attention/global/qkv_kernel:0'
    self.assertIn(test_variable_name1, variable_names)
    self.assertIn(test_variable_name2, variable_names)


if __name__ == '__main__':
  tf.test.main()
