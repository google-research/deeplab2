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

"""This file contains the code for the Motion-DeepLab decoder."""

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.decoder import panoptic_deeplab


class MotionDeepLabDecoder(tf.keras.layers.Layer):
  """A Motion-DeepLab decoder layer.

  This layer takes low- and high-level features as input and uses a dual-ASPP
  and dual-decoder structure to aggregate features for semantic and instance
  segmentation. On top of the decoders, four heads are used to predict semantic
  segmentation, instance center probabilities, instance center regression, and
  previous frame offset regression per pixel.
  """

  def __init__(
      self,
      decoder_options: config_pb2.DecoderOptions,
      motion_deeplab_options: config_pb2.ModelOptions.MotionDeepLabOptions,
      bn_layer=tf.keras.layers.BatchNormalization):
    """Initializes a Motion-DeepLab decoder.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      motion_deeplab_options: Model options as defined in
        config_pb2.ModelOptions.MotionDeeplabOptions.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    """
    super(MotionDeepLabDecoder, self).__init__(name='MotionDeepLabDecoder')

    low_level_feature_keys = [
        item.feature_key for item in motion_deeplab_options.low_level
    ]
    low_level_channels_project = [
        item.channels_project for item in motion_deeplab_options.low_level
    ]

    self._semantic_decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        decoder_options.feature_key,
        low_level_feature_keys,
        low_level_channels_project,
        decoder_options.aspp_channels,
        decoder_options.decoder_channels,
        decoder_options.atrous_rates,
        name='semantic_decoder',
        bn_layer=bn_layer)
    self._semantic_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        motion_deeplab_options.semantic_head.head_channels,
        motion_deeplab_options.semantic_head.output_channels,
        common.PRED_SEMANTIC_LOGITS_KEY,
        name='semantic_head',
        bn_layer=bn_layer)

    self._instance_decoder = None
    self._instance_center_head = None
    self._instance_regression_head = None
    self._motion_regression_head = None

    if motion_deeplab_options.instance.low_level_override:
      low_level_options = motion_deeplab_options.instance.low_level_override
    else:
      low_level_options = motion_deeplab_options.low_level

    # If instance_decoder is set, use those options; otherwise reuse the
    # architecture as defined for the semantic decoder.
    if motion_deeplab_options.instance.HasField('instance_decoder_override'):
      decoder_options = (motion_deeplab_options.instance
                         .instance_decoder_override)

    low_level_feature_keys = [item.feature_key for item in low_level_options]
    low_level_channels_project = [
        item.channels_project for item in low_level_options
    ]

    self._instance_decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        decoder_options.feature_key,
        low_level_feature_keys,
        low_level_channels_project,
        decoder_options.aspp_channels,
        decoder_options.decoder_channels,
        decoder_options.atrous_rates,
        name='instance_decoder',
        bn_layer=bn_layer)
    self._instance_center_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        motion_deeplab_options.instance.center_head.head_channels,
        motion_deeplab_options.instance.center_head.output_channels,
        common.PRED_CENTER_HEATMAP_KEY,
        name='instance_center_head',
        bn_layer=bn_layer)
    self._instance_regression_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        motion_deeplab_options.instance.regression_head.head_channels,
        motion_deeplab_options.instance.regression_head.output_channels,
        common.PRED_OFFSET_MAP_KEY,
        name='instance_regression_head',
        bn_layer=bn_layer)

    # The motion head regresses every pixel to its center in the previous
    # frame.
    self._motion_regression_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        motion_deeplab_options.motion_head.head_channels,
        motion_deeplab_options.motion_head.output_channels,
        common.PRED_FRAME_OFFSET_MAP_KEY,
        name='motion_regression_head',
        bn_layer=bn_layer)

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layers to global average pooling."""
    self._semantic_decoder.reset_pooling_layer()
    if self._instance_decoder is not None:
      self._instance_decoder.reset_pooling_layer()

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layers.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layers.
    """
    self._semantic_decoder.set_pool_size(pool_size)
    if self._instance_decoder is not None:
      self._instance_decoder.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._semantic_decoder.get_pool_size()

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels]. Different keys should point to different features extracted
        by the encoder, e.g. low-level or high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the results of the semantic segmentation head and
        depending on the configuration also of the instance segmentation head.
    """

    semantic_features = self._semantic_decoder(features, training=training)
    results = self._semantic_head(semantic_features, training=training)

    if self._instance_decoder is not None:
      instance_features = self._instance_decoder(features, training=training)
      instance_center_predictions = self._instance_center_head(
          instance_features, training=training)
      instance_regression_predictions = self._instance_regression_head(
          instance_features, training=training)
      motion_regression_predictions = self._motion_regression_head(
          instance_features, training=training)
      if results.keys() & motion_regression_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'motion branch overlap. Please use unique keys.')
      results.update(motion_regression_predictions)

      if results.keys() & instance_center_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'center branch overlap. Please use unique keys.')
      results.update(instance_center_predictions)

      if results.keys() & instance_regression_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'regression branch overlap. Please use unique keys.')
      results.update(instance_regression_predictions)

    return results

  @property
  def checkpoint_items(self):
    items = {
        common.CKPT_SEMANTIC_DECODER:
            self._semantic_decoder,
        common.CKPT_SEMANTIC_HEAD_WITHOUT_LAST_LAYER:
            self._semantic_head.conv_block,
        common.CKPT_SEMANTIC_LAST_LAYER:
            self._semantic_head.final_conv
    }
    if self._instance_decoder is not None:
      instance_items = {
          common.CKPT_INSTANCE_DECODER:
              self._instance_decoder,
          common.CKPT_INSTANCE_CENTER_HEAD_WITHOUT_LAST_LAYER:
              self._instance_center_head.conv_block,
          common.CKPT_INSTANCE_CENTER_HEAD_LAST_LAYER:
              self._instance_center_head.final_conv,
          common.CKPT_INSTANCE_REGRESSION_HEAD_WITHOUT_LAST_LAYER:
              self._instance_regression_head.conv_block,
          common.CKPT_INSTANCE_REGRESSION_HEAD_LAST_LAYER:
              self._instance_regression_head.final_conv,
          common.CKPT_MOTION_REGRESSION_HEAD_WITHOUT_LAST_LAYER:
              self._motion_regression_head.conv_block,
          common.CKPT_MOTION_REGRESSION_HEAD_LAST_LAYER:
              self._motion_regression_head.final_conv,
      }
      items.update(instance_items)
    return items
