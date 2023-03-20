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

"""This file contains code to build a DeepLabV3.

Reference:
  - [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
"""
import tensorflow as tf

from deeplab2 import common
from deeplab2.model.decoder import aspp
from deeplab2.model.layers import convolutions


layers = tf.keras.layers


class DeepLabV3(layers.Layer):
  """A DeepLabV3 model.

  This model takes in features from an encoder and performs multi-scale context
  aggregation with the help of an ASPP layer. Finally, a classification head is
  used to predict a semantic segmentation.
  """

  def __init__(self,
               decoder_options,
               deeplabv3_options,
               bn_layer=tf.keras.layers.BatchNormalization):
    """Creates a DeepLabV3 decoder of type layers.Layer.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      deeplabv3_options: Model options as defined in
        config_pb2.ModelOptions.DeeplabV3Options.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    """
    super(DeepLabV3, self).__init__(name='DeepLabV3')

    self._feature_name = decoder_options.feature_key
    self._aspp = aspp.ASPP(decoder_options.aspp_channels,
                           decoder_options.atrous_rates,
                           bn_layer=bn_layer)

    self._classifier_conv_bn_act = convolutions.Conv2DSame(
        decoder_options.decoder_channels,
        kernel_size=3,
        name='classifier_conv_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation='relu')

    self._final_conv = convolutions.Conv2DSame(
        deeplabv3_options.num_classes, kernel_size=1, name='final_conv')

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layer.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    self._aspp.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._aspp.get_pool_size()

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layer to global average pooling."""
    self._aspp.reset_pooling_layer()

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: A single input tf.Tensor or an input dict of tf.Tensor with
        shape [batch, height, width, channels]. If passed a dict, different keys
        should point to different features extracted by the encoder, e.g.
        low-level or high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the semantic prediction under key
      common.PRED_SEMANTIC_LOGITS_KEY.
    """
    if isinstance(features, tf.Tensor):
      feature = features
    else:
      feature = features[self._feature_name]

    x = self._aspp(feature, training=training)

    x = self._classifier_conv_bn_act(x, training=training)

    return {common.PRED_SEMANTIC_LOGITS_KEY: self._final_conv(x)}

  @property
  def checkpoint_items(self):
    items = {
        common.CKPT_DEEPLABV3_ASPP: self._aspp,
        common.CKPT_DEEPLABV3_CLASSIFIER_CONV_BN_ACT:
            self._classifier_conv_bn_act,
        common.CKPT_SEMANTIC_LAST_LAYER: self._final_conv,
    }
    return items
