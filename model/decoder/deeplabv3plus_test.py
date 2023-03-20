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

"""Tests for deeplabv3plus."""

import numpy as np
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.decoder import deeplabv3plus
from deeplab2.utils import test_utils


def _create_deeplabv3plus_model(high_level_feature_name, low_level_feature_name,
                                low_level_channels_project,
                                aspp_output_channels, decoder_output_channels,
                                atrous_rates, num_classes, **kwargs):
  decoder_options = config_pb2.DecoderOptions(
      feature_key=high_level_feature_name,
      decoder_channels=decoder_output_channels,
      aspp_channels=aspp_output_channels,
      atrous_rates=atrous_rates)
  deeplabv3plus_options = config_pb2.ModelOptions.DeeplabV3PlusOptions(
      low_level=config_pb2.LowLevelOptions(
          feature_key=low_level_feature_name,
          channels_project=low_level_channels_project),
      num_classes=num_classes)
  return deeplabv3plus.DeepLabV3Plus(decoder_options, deeplabv3plus_options,
                                     **kwargs)


class Deeplabv3PlusTest(tf.test.TestCase):

  def test_deeplabv3plus_feature_key_not_present(self):
    deeplabv3plus_decoder = _create_deeplabv3plus_model(
        high_level_feature_name='not_in_features_dict',
        low_level_feature_name='in_feature_dict',
        low_level_channels_project=128,
        aspp_output_channels=64,
        decoder_output_channels=64,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    input_dict = dict()
    input_dict['in_feature_dict'] = tf.random.uniform(shape=(2, 65, 65, 32))

    with self.assertRaises(KeyError):
      _ = deeplabv3plus_decoder(input_dict)

  def test_deeplabv3plus_output_shape(self):
    list_of_num_classes = [2, 19, 133]
    for num_classes in list_of_num_classes:
      deeplabv3plus_decoder = _create_deeplabv3plus_model(
          high_level_feature_name='high',
          low_level_feature_name='low',
          low_level_channels_project=128,
          aspp_output_channels=64,
          decoder_output_channels=128,
          atrous_rates=[6, 12, 18],
          num_classes=num_classes)
      input_dict = dict()
      input_dict['high'] = tf.random.uniform(shape=(2, 65, 65, 32))
      input_dict['low'] = tf.random.uniform(shape=(2, 129, 129, 16))
      expected_shape = [2, 129, 129, num_classes]

      logit_tensor = deeplabv3plus_decoder(input_dict)
      self.assertListEqual(
          logit_tensor[common.PRED_SEMANTIC_LOGITS_KEY].shape.as_list(),
          expected_shape)

  def test_deeplabv3plus_feature_extraction_consistency(self):
    deeplabv3plus_decoder = _create_deeplabv3plus_model(
        high_level_feature_name='high',
        low_level_feature_name='low',
        low_level_channels_project=128,
        aspp_output_channels=96,
        decoder_output_channels=64,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    input_dict = dict()
    input_dict['high'] = tf.random.uniform(shape=(2, 65, 65, 32))
    input_dict['low'] = tf.random.uniform(shape=(2, 129, 129, 16))

    reference_logits_tensor = deeplabv3plus_decoder(
        input_dict, training=False)
    logits_tensor_to_compare = deeplabv3plus_decoder(input_dict, training=False)

    np.testing.assert_equal(
        reference_logits_tensor[common.PRED_SEMANTIC_LOGITS_KEY].numpy(),
        logits_tensor_to_compare[common.PRED_SEMANTIC_LOGITS_KEY].numpy())

  def test_deeplabv3plus_pool_size_setter(self):
    deeplabv3plus_decoder = _create_deeplabv3plus_model(
        high_level_feature_name='high',
        low_level_feature_name='low',
        low_level_channels_project=128,
        aspp_output_channels=96,
        decoder_output_channels=64,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    pool_size = (10, 10)
    deeplabv3plus_decoder.set_pool_size(pool_size)

    self.assertTupleEqual(deeplabv3plus_decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  @test_utils.test_all_strategies
  def test_deeplabv3plus_sync_bn(self, strategy):
    input_dict = dict()
    input_dict['high'] = tf.random.uniform(shape=(2, 65, 65, 32))
    input_dict['low'] = tf.random.uniform(shape=(2, 129, 129, 16))
    with strategy.scope():
      for bn_layer in test_utils.NORMALIZATION_LAYERS:
        deeplabv3plus_decoder = _create_deeplabv3plus_model(
            high_level_feature_name='high',
            low_level_feature_name='low',
            low_level_channels_project=128,
            aspp_output_channels=96,
            decoder_output_channels=64,
            atrous_rates=[6, 12, 18],
            num_classes=80,
            bn_layer=bn_layer)
        _ = deeplabv3plus_decoder(input_dict)

  def test_deeplabv3plus_pool_size_resetter(self):
    deeplabv3plus_decoder = _create_deeplabv3plus_model(
        high_level_feature_name='high',
        low_level_feature_name='low',
        low_level_channels_project=128,
        aspp_output_channels=96,
        decoder_output_channels=64,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    pool_size = (None, None)
    deeplabv3plus_decoder.reset_pooling_layer()

    self.assertTupleEqual(deeplabv3plus_decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  def test_deeplabv3plus_ckpt_items(self):
    deeplabv3plus_decoder = _create_deeplabv3plus_model(
        high_level_feature_name='high',
        low_level_feature_name='low',
        low_level_channels_project=128,
        aspp_output_channels=96,
        decoder_output_channels=64,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    ckpt_dict = deeplabv3plus_decoder.checkpoint_items
    self.assertIn(common.CKPT_DEEPLABV3PLUS_ASPP, ckpt_dict)
    self.assertIn(common.CKPT_DEEPLABV3PLUS_PROJECT_CONV_BN_ACT, ckpt_dict)
    self.assertIn(common.CKPT_DEEPLABV3PLUS_FUSE, ckpt_dict)
    self.assertIn(common.CKPT_SEMANTIC_LAST_LAYER, ckpt_dict)


if __name__ == '__main__':
  tf.test.main()
