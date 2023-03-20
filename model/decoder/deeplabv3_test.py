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

"""Tests for deeplabv3."""

import numpy as np
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.decoder import deeplabv3
from deeplab2.utils import test_utils


def _create_deeplabv3_model(feature_key, decoder_channels, aspp_channels,
                            atrous_rates, num_classes, **kwargs):
  decoder_options = config_pb2.DecoderOptions(
      feature_key=feature_key,
      decoder_channels=decoder_channels,
      aspp_channels=aspp_channels,
      atrous_rates=atrous_rates)
  deeplabv3_options = config_pb2.ModelOptions.DeeplabV3Options(
      num_classes=num_classes)
  return deeplabv3.DeepLabV3(decoder_options, deeplabv3_options, **kwargs)


class Deeplabv3Test(tf.test.TestCase):

  def test_deeplabv3_feature_key_not_present(self):
    deeplabv3_decoder = _create_deeplabv3_model(
        feature_key='not_in_features_dict',
        aspp_channels=64,
        decoder_channels=48,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    input_dict = dict()
    input_dict['not_the_same_key'] = tf.random.uniform(shape=(2, 65, 65, 32))

    with self.assertRaises(KeyError):
      _ = deeplabv3_decoder(input_dict)

  def test_deeplabv3_output_shape(self):
    list_of_num_classes = [2, 19, 133]
    for num_classes in list_of_num_classes:
      deeplabv3_decoder = _create_deeplabv3_model(
          feature_key='not_used',
          aspp_channels=64,
          decoder_channels=48,
          atrous_rates=[6, 12, 18],
          num_classes=num_classes)
      input_tensor = tf.random.uniform(shape=(2, 65, 65, 32))
      expected_shape = [2, 65, 65, num_classes]

      logit_tensor = deeplabv3_decoder(input_tensor)
      self.assertListEqual(
          logit_tensor[common.PRED_SEMANTIC_LOGITS_KEY].shape.as_list(),
          expected_shape)

  @test_utils.test_all_strategies
  def test_sync_bn(self, strategy):
    input_tensor = tf.random.uniform(shape=(2, 65, 65, 32))
    with strategy.scope():
      for bn_layer in test_utils.NORMALIZATION_LAYERS:
        deeplabv3_decoder = _create_deeplabv3_model(
            feature_key='not_used',
            aspp_channels=64,
            decoder_channels=48,
            atrous_rates=[6, 12, 18],
            num_classes=19,
            bn_layer=bn_layer)
        _ = deeplabv3_decoder(input_tensor)

  def test_deeplabv3_feature_extraction_consistency(self):
    deeplabv3_decoder = _create_deeplabv3_model(
        aspp_channels=64,
        decoder_channels=48,
        atrous_rates=[6, 12, 18],
        num_classes=80,
        feature_key='feature_key')
    input_tensor = tf.random.uniform(shape=(2, 65, 65, 32))
    input_dict = dict()
    input_dict['feature_key'] = input_tensor

    reference_logits_tensor = deeplabv3_decoder(input_tensor, training=False)
    logits_tensor_to_compare = deeplabv3_decoder(input_dict, training=False)

    np.testing.assert_equal(
        reference_logits_tensor[common.PRED_SEMANTIC_LOGITS_KEY].numpy(),
        logits_tensor_to_compare[common.PRED_SEMANTIC_LOGITS_KEY].numpy())

  def test_deeplabv3_pool_size_setter(self):
    deeplabv3_decoder = _create_deeplabv3_model(
        feature_key='not_used',
        aspp_channels=64,
        decoder_channels=48,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    pool_size = (10, 10)
    deeplabv3_decoder.set_pool_size(pool_size)

    self.assertTupleEqual(deeplabv3_decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  def test_deeplabv3_pool_size_resetter(self):
    deeplabv3_decoder = _create_deeplabv3_model(
        feature_key='not_used',
        aspp_channels=64,
        decoder_channels=48,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    pool_size = (None, None)
    deeplabv3_decoder.reset_pooling_layer()

    self.assertTupleEqual(deeplabv3_decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  def test_deeplabv3_ckpt_items(self):
    deeplabv3_decoder = _create_deeplabv3_model(
        feature_key='not_used',
        aspp_channels=64,
        decoder_channels=48,
        atrous_rates=[6, 12, 18],
        num_classes=80)
    ckpt_dict = deeplabv3_decoder.checkpoint_items
    self.assertIn(common.CKPT_DEEPLABV3_ASPP, ckpt_dict)
    self.assertIn(common.CKPT_DEEPLABV3_CLASSIFIER_CONV_BN_ACT, ckpt_dict)
    self.assertIn(common.CKPT_SEMANTIC_LAST_LAYER, ckpt_dict)


if __name__ == '__main__':
  tf.test.main()
