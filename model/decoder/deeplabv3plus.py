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

"""This file contains code to build a DeepLabV3Plus.

Reference:
  - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
      Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
"""

import tensorflow as tf

from deeplab2 import common
from deeplab2.model import utils
from deeplab2.model.decoder import aspp
from deeplab2.model.layers import convolutions


layers = tf.keras.layers


class DeepLabV3Plus(tf.keras.layers.Layer):
  """A DeepLabV3+ decoder model.

  This model takes in low- and high-level features from an encoder and performs
  multi-scale context aggregation with the help of an ASPP layer on high-level
  features. These are concatenated with the low-level features and used as input
  to the classification head that is used to predict a semantic segmentation.
  """

  def __init__(self,
               decoder_options,
               deeplabv3plus_options,
               bn_layer=tf.keras.layers.BatchNormalization):
    """Creates a DeepLabV3+ decoder of type tf.keras.layers.Layer.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      deeplabv3plus_options: Model options as defined in
        config_pb2.ModelOptions.DeeplabV3PlusOptions.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    """
    super(DeepLabV3Plus, self).__init__(name='DeepLabv3Plus')

    self._high_level_feature_name = decoder_options.feature_key
    self._low_level_feature_name = deeplabv3plus_options.low_level.feature_key
    self._aspp = aspp.ASPP(decoder_options.aspp_channels,
                           decoder_options.atrous_rates,
                           bn_layer=bn_layer)

    # Layers for low-level feature transformation.
    self._project_conv_bn_act = convolutions.Conv2DSame(
        deeplabv3plus_options.low_level.channels_project,
        kernel_size=1,
        name='project_conv_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation='relu')

    # Layers for fusing low- and high-level features.
    self._fuse = convolutions.StackedConv2DSame(
        conv_type='depthwise_separable_conv',
        num_layers=2,
        output_channels=decoder_options.decoder_channels,
        kernel_size=3,
        name='fuse',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation='relu')

    self._final_conv = convolutions.Conv2DSame(
        deeplabv3plus_options.num_classes, kernel_size=1, name='final_conv')

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layer to global average pooling."""
    self._aspp.reset_pooling_layer()

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layer.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    self._aspp.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._aspp.get_pool_size()

  @property
  def checkpoint_items(self):
    items = {
        common.CKPT_DEEPLABV3PLUS_ASPP: self._aspp,
        common.CKPT_DEEPLABV3PLUS_PROJECT_CONV_BN_ACT:
            self._project_conv_bn_act,
        common.CKPT_DEEPLABV3PLUS_FUSE: self._fuse,
        common.CKPT_SEMANTIC_LAST_LAYER: self._final_conv,
    }
    return items

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels]. Different keys should point to different features extracted
        by the encoder, e.g. low-level or high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the semantic prediction under key
      common.PRED_SEMANTIC_LOGITS_KEY.
    """
    low_level_features = features[self._low_level_feature_name]
    high_level_features = features[self._high_level_feature_name]

    high_level_features = self._aspp(high_level_features, training=training)

    low_level_features = self._project_conv_bn_act(low_level_features,
                                                   training=training)

    target_h = tf.shape(low_level_features)[1]
    target_w = tf.shape(low_level_features)[2]

    high_level_features = utils.resize_align_corners(
        high_level_features, [target_h, target_w])
    x = tf.concat([high_level_features, low_level_features], 3)
    x = self._fuse(x)

    return {common.PRED_SEMANTIC_LOGITS_KEY: self._final_conv(x)}
