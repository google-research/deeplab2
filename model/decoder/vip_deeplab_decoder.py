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

"""This file contains code to build a ViP-DeepLab decoder.

Reference:
  - [ViP-DeepLab: Learning Visual Perception with Depth-aware Video
      Panoptic Segmentation](https://arxiv.org/abs/2012.05258)
"""
import tensorflow as tf

from deeplab2 import common
from deeplab2.model import utils
from deeplab2.model.decoder import panoptic_deeplab

layers = tf.keras.layers


class ViPDeepLabDecoder(layers.Layer):
  """A ViP-DeepLab decoder layer.

  This layer takes low- and high-level features as input and uses a dual-ASPP
  and dual-decoder structure to aggregate features for semantic and instance
  segmentation. On top of the decoders, three heads are used to predict semantic
  segmentation, instance center probabilities, and instance center regression
  per pixel. It also has a branch to predict the next-frame instance center
  regression. Different from the ViP-DeepLab paper which uses Cascade-ASPP, this
  reimplementation only uses ASPP.
  """

  def __init__(self,
               decoder_options,
               vip_deeplab_options,
               bn_layer=tf.keras.layers.BatchNormalization):
    """Initializes a ViP-DeepLab decoder.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      vip_deeplab_options: Model options as defined in
        config_pb2.ModelOptions.ViPDeeplabOptions.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    """
    super(ViPDeepLabDecoder, self).__init__(name='ViPDeepLab')

    low_level_feature_keys = [
        item.feature_key for item in vip_deeplab_options.low_level
    ]
    low_level_channels_project = [
        item.channels_project for item in vip_deeplab_options.low_level
    ]

    self._semantic_decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        high_level_feature_name=decoder_options.feature_key,
        low_level_feature_names=low_level_feature_keys,
        low_level_channels_project=low_level_channels_project,
        aspp_output_channels=decoder_options.aspp_channels,
        decoder_output_channels=decoder_options.decoder_channels,
        atrous_rates=decoder_options.atrous_rates,
        name='semantic_decoder',
        aspp_use_only_1x1_proj_conv=decoder_options.aspp_use_only_1x1_proj_conv,
        decoder_conv_type=decoder_options.decoder_conv_type,
        bn_layer=bn_layer)
    self._semantic_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        vip_deeplab_options.semantic_head.head_channels,
        vip_deeplab_options.semantic_head.output_channels,
        common.PRED_SEMANTIC_LOGITS_KEY,
        name='semantic_head',
        conv_type=vip_deeplab_options.semantic_head.head_conv_type,
        bn_layer=bn_layer)

    self._depth_head = None
    if vip_deeplab_options.HasField('depth_head'):
      self._depth_head = panoptic_deeplab.PanopticDeepLabSingleHead(
          vip_deeplab_options.depth_head.head_channels,
          vip_deeplab_options.depth_head.output_channels,
          common.PRED_DEPTH_KEY,
          name='depth_head',
          conv_type=vip_deeplab_options.depth_head.head_conv_type,
          bn_layer=bn_layer)
      self._max_depth = (
          vip_deeplab_options.depth_head.max_value_after_activation)
      self._min_depth = (
          vip_deeplab_options.depth_head.min_value_after_activation)

    self._instance_decoder = None
    self._instance_center_head = None
    self._instance_regression_head = None
    self._next_instance_decoder = None
    self._next_instance_regression_head = None

    if vip_deeplab_options.instance.enable:
      if vip_deeplab_options.instance.low_level_override:
        low_level_options = vip_deeplab_options.instance.low_level_override
      else:
        low_level_options = vip_deeplab_options.low_level

      # If instance_decoder is set, use those options; otherwise reuse the
      # architecture as defined for the semantic decoder.
      if vip_deeplab_options.instance.HasField('instance_decoder_override'):
        decoder_options = (
            vip_deeplab_options.instance.instance_decoder_override)

      low_level_feature_keys = [item.feature_key for item in low_level_options]
      low_level_channels_project = [
          item.channels_project for item in low_level_options
      ]

      self._instance_decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
          high_level_feature_name=decoder_options.feature_key,
          low_level_feature_names=low_level_feature_keys,
          low_level_channels_project=low_level_channels_project,
          aspp_output_channels=decoder_options.aspp_channels,
          decoder_output_channels=decoder_options.decoder_channels,
          atrous_rates=decoder_options.atrous_rates,
          name='instance_decoder',
          aspp_use_only_1x1_proj_conv=(
              decoder_options.aspp_use_only_1x1_proj_conv),
          decoder_conv_type=decoder_options.decoder_conv_type,
          bn_layer=bn_layer)
      self._instance_center_head = panoptic_deeplab.PanopticDeepLabSingleHead(
          vip_deeplab_options.instance.center_head.head_channels,
          vip_deeplab_options.instance.center_head.output_channels,
          common.PRED_CENTER_HEATMAP_KEY,
          name='instance_center_head',
          conv_type=(vip_deeplab_options.instance.center_head.head_conv_type),
          bn_layer=bn_layer)
      self._instance_regression_head = (
          panoptic_deeplab.PanopticDeepLabSingleHead(
              vip_deeplab_options.instance.regression_head.head_channels,
              vip_deeplab_options.instance.regression_head.output_channels,
              common.PRED_OFFSET_MAP_KEY,
              name='instance_regression_head',
              conv_type=(
                  vip_deeplab_options.instance.regression_head.head_conv_type),
              bn_layer=bn_layer))

      if vip_deeplab_options.instance.HasField('next_regression_head'):
        self._next_instance_decoder = (
            panoptic_deeplab.PanopticDeepLabSingleDecoder(
                high_level_feature_name=decoder_options.feature_key,
                low_level_feature_names=low_level_feature_keys,
                low_level_channels_project=low_level_channels_project,
                aspp_output_channels=decoder_options.aspp_channels,
                decoder_output_channels=decoder_options.decoder_channels,
                atrous_rates=decoder_options.atrous_rates,
                name='next_instance_decoder',
                aspp_use_only_1x1_proj_conv=(
                    decoder_options.aspp_use_only_1x1_proj_conv),
                decoder_conv_type=decoder_options.decoder_conv_type,
                bn_layer=bn_layer))
        self._next_instance_regression_head = (
            panoptic_deeplab.PanopticDeepLabSingleHead(
                (vip_deeplab_options.instance.next_regression_head.head_channels
                ), (vip_deeplab_options.instance.next_regression_head
                    .output_channels),
                common.PRED_NEXT_OFFSET_MAP_KEY,
                name='next_instance_regression_head',
                conv_type=(vip_deeplab_options.instance.next_regression_head
                           .head_conv_type),
                bn_layer=bn_layer))
        self._next_high_level_feature_name = decoder_options.feature_key

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layers to global average pooling."""
    self._semantic_decoder.reset_pooling_layer()
    if self._instance_decoder is not None:
      self._instance_decoder.reset_pooling_layer()
    if self._next_instance_decoder is not None:
      self._next_instance_decoder.reset_pooling_layer()

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layers.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layers.
    """
    self._semantic_decoder.set_pool_size(pool_size)
    if self._instance_decoder is not None:
      self._instance_decoder.set_pool_size(pool_size)
    if self._next_instance_decoder is not None:
      self._next_instance_decoder.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._semantic_decoder.get_pool_size()

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
      }
      items.update(instance_items)
    if self._next_instance_decoder is not None:
      next_instance_items = {
          common.CKPT_NEXT_INSTANCE_DECODER:
              self._next_instance_decoder,
          common.CKPT_NEXT_INSTANCE_REGRESSION_HEAD_WITHOUT_LAST_LAYER:
              self._next_instance_regression_head.conv_block,
          common.CKPT_NEXT_INSTANCE_REGRESSION_HEAD_LAST_LAYER:
              self._next_instance_regression_head.final_conv,
      }
      items.update(next_instance_items)
    if self._depth_head is not None:
      depth_items = {
          common.CKPT_DEPTH_HEAD_WITHOUT_LAST_LAYER:
              self._depth_head.conv_block,
          common.CKPT_DEPTH_HEAD_LAST_LAYER:
              self._depth_head.final_conv
      }
      items.update(depth_items)
    return items

  def call(self, features, next_features, training=False):
    """Performs a forward pass.

    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels]. Different keys should point to different features extracted
        by the encoder, e.g. low-level or high-level features.
      next_features: An input dict of tf.Tensor similar to features. The
        features are computed with the next frame as input.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the results of the semantic segmentation head and
        depending on the configuration also of the instance segmentation head.
    """

    semantic_features = self._semantic_decoder(features, training=training)
    results = self._semantic_head(semantic_features, training=training)

    if self._depth_head is not None:
      feature_size = semantic_features.get_shape().as_list()[1:3]
      scaled_feature_size = utils.scale_mutable_sequence(feature_size, 2)
      depth_features = utils.resize_align_corners(semantic_features,
                                                  scaled_feature_size)
      depth_prediction = self._depth_head(depth_features)
      for pred_key, pred_value in depth_prediction.items():
        pred_value = self._min_depth + tf.sigmoid(pred_value) * (
            self._max_depth - self._min_depth)
        depth_prediction[pred_key] = pred_value
      results.update(depth_prediction)

    if self._instance_decoder is not None:
      instance_features = self._instance_decoder(features, training=training)
      instance_center_predictions = self._instance_center_head(
          instance_features, training=training)
      instance_regression_predictions = self._instance_regression_head(
          instance_features, training=training)

      if results.keys() & instance_center_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'center branch overlap. Please use unique keys.')
      results.update(instance_center_predictions)

      if results.keys() & instance_regression_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'regression branch overlap. Please use unique keys.')
      results.update(instance_regression_predictions)

    if self._next_instance_decoder is not None:
      # We update the high level features in next_features with the concated
      # features of the high level features in both features and next_features.
      high_level_feature_name = self._next_high_level_feature_name
      high_level_features = features[high_level_feature_name]
      next_high_level_features = next_features[high_level_feature_name]
      next_high_level_features = tf.concat(
          [high_level_features, next_high_level_features], axis=3)
      # Create a new dict for next_features to keep the original next_features
      # untouched for model exporting.
      new_next_features = dict()
      for key in next_features:
        if key == high_level_feature_name:
          new_next_features[key] = next_high_level_features
        else:
          new_next_features[key] = next_features[key]
      next_regression_features = self._next_instance_decoder(
          new_next_features, training=training)
      next_regression_predictions = self._next_instance_regression_head(
          next_regression_features, training=training)
      if results.keys() & next_regression_predictions.keys():
        raise ValueError('The keys of the next regresion branch overlap.'
                         'Please use unique keys.')
      results.update(next_regression_predictions)

    return results
