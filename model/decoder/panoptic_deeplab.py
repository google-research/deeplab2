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

"""This file contains code to build a Panoptic-DeepLab decoder.

Reference:
  - [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up
      Panoptic Segmentation](https://arxiv.org/pdf/1911.10194)
"""
from absl import logging

import tensorflow as tf

from deeplab2 import common
from deeplab2.model import utils
from deeplab2.model.decoder import aspp
from deeplab2.model.layers import convolutions


layers = tf.keras.layers


class PanopticDeepLabSingleDecoder(layers.Layer):
  """A single Panoptic-DeepLab decoder layer.

  This layer takes low- and high-level features as input and uses an ASPP
  followed by a fusion block to decode features for a single task, e.g.,
  semantic segmentation or instance segmentation.
  """

  def __init__(self,
               high_level_feature_name,
               low_level_feature_names,
               low_level_channels_project,
               aspp_output_channels,
               decoder_output_channels,
               atrous_rates,
               name,
               aspp_use_only_1x1_proj_conv=False,
               decoder_conv_type='depthwise_separable_conv',
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Initializes a single Panoptic-DeepLab decoder of layers.Layer.

    Args:
      high_level_feature_name: A string specifying the name of the high-level
        feature coming from an encoder.
      low_level_feature_names: A list of strings specifying the name of the
        low-level features coming from an encoder. An order from highest to
        lower level is expected, e.g. ['res3', 'res2'].
      low_level_channels_project: A list of integer specifying the number of
        filters used for processing each low_level features.
      aspp_output_channels: An integer specifying the number of filters in the
        ASPP convolution layers.
      decoder_output_channels: An integer specifying the number of filters in
        the decoder convolution layers.
      atrous_rates: A list of three integers specifying the atrous rate for the
        ASPP layers.
      name: A string specifying the name of the layer.
      aspp_use_only_1x1_proj_conv: Boolean, specifying if the ASPP five branches
        are turned off or not. If True, the ASPP module is degenerated to one
        1x1 convolution, projecting the input channels to `output_channels`.
      decoder_conv_type: String, specifying decoder convolution type. Support
        'depthwise_separable_conv' and 'standard_conv'.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.

    Raises:
      ValueError: An error occurs when the length of low_level_feature_names
        differs from the length of low_level_channels_project.
    """
    super(PanopticDeepLabSingleDecoder, self).__init__(name=name)
    self._channel_axis = 3

    self._aspp = aspp.ASPP(
        aspp_output_channels,
        atrous_rates,
        aspp_use_only_1x1_proj_conv=aspp_use_only_1x1_proj_conv,
        name='aspp',
        bn_layer=bn_layer,
        activation=activation)
    self._high_level_feature_name = high_level_feature_name

    if len(low_level_feature_names) != len(low_level_channels_project):
      raise ValueError('The Panoptic-DeepLab decoder requires the same number '
                       'of low-level features as the number of low-level '
                       'projection channels. But got %d and %d.'
                       % (len(low_level_feature_names),
                          len(low_level_channels_project)))

    self._low_level_feature_names = low_level_feature_names

    for i, channels_project in enumerate(low_level_channels_project):
      # Check if channel sizes increases and issue a warning.
      if i > 0 and low_level_channels_project[i - 1] < channels_project:
        logging.warning(
            'The low level projection channels usually do not '
            'increase for features with higher spatial resolution. '
            'Please make sure, this behavior is intended.')
      current_low_level_conv_name, current_fusion_conv_name = (
          utils.get_low_level_conv_fusion_conv_current_names(i))
      utils.safe_setattr(
          self, current_low_level_conv_name, convolutions.Conv2DSame(
              channels_project,
              kernel_size=1,
              name=utils.get_layer_name(current_low_level_conv_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation=activation))

      utils.safe_setattr(
          self, current_fusion_conv_name, convolutions.StackedConv2DSame(
              conv_type=decoder_conv_type,
              num_layers=1,
              output_channels=decoder_output_channels,
              kernel_size=5,
              name=utils.get_layer_name(current_fusion_conv_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation=activation))

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels]. Different keys should point to different features extracted
        by the encoder, e.g. low-level or high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      Refined features as instance of tf.Tensor.
    """

    high_level_features = features[self._high_level_feature_name]
    combined_features = self._aspp(high_level_features, training=training)

    # Fuse low-level features with high-level features.
    for i in range(len(self._low_level_feature_names)):
      current_low_level_conv_name, current_fusion_conv_name = (
          utils.get_low_level_conv_fusion_conv_current_names(i))
      # Iterate from the highest level of the low level features to the lowest
      # level, i.e. take the features with the smallest spatial size first.
      low_level_features = features[self._low_level_feature_names[i]]
      low_level_features = getattr(self, current_low_level_conv_name)(
          low_level_features, training=training)

      target_h = tf.shape(low_level_features)[1]
      target_w = tf.shape(low_level_features)[2]
      source_h = tf.shape(combined_features)[1]
      source_w = tf.shape(combined_features)[2]

      tf.assert_less(
          source_h - 1,
          target_h,
          message='Features are down-sampled during decoder.')
      tf.assert_less(
          source_w - 1,
          target_w,
          message='Features are down-sampled during decoder.')

      combined_features = utils.resize_align_corners(combined_features,
                                                     [target_h, target_w])

      combined_features = tf.concat([combined_features, low_level_features],
                                    self._channel_axis)
      combined_features = getattr(self, current_fusion_conv_name)(
          combined_features, training=training)

    return combined_features

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


class PanopticDeepLabSingleHead(layers.Layer):
  """A single PanopticDeepLab head layer.

  This layer takes in the enriched features from a decoder and adds two
  convolutions on top.
  """

  def __init__(self,
               intermediate_channels,
               output_channels,
               pred_key,
               name,
               conv_type='depthwise_separable_conv',
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Initializes a single PanopticDeepLab head.

    Args:
      intermediate_channels: An integer specifying the number of filters of the
        first 5x5 convolution.
      output_channels: An integer specifying the number of filters of the second
        1x1 convolution.
      pred_key: A string specifying the key of the output dictionary.
      name: A string specifying the name of this head.
      conv_type: String, specifying head convolution type. Support
        'depthwise_separable_conv' and 'standard_conv'.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
    """
    super(PanopticDeepLabSingleHead, self).__init__(name=name)
    self._pred_key = pred_key

    self.conv_block = convolutions.StackedConv2DSame(
        conv_type=conv_type,
        num_layers=1,
        output_channels=intermediate_channels,
        kernel_size=5,
        name='conv_block',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation)
    self.final_conv = layers.Conv2D(
        output_channels,
        kernel_size=1,
        name='final_conv',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: A tf.Tensor with shape [batch, height, width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The dictionary containing the predictions under the specified key.
    """
    x = self.conv_block(features, training=training)
    return {self._pred_key: self.final_conv(x)}


class PanopticDeepLab(layers.Layer):
  """A Panoptic-DeepLab decoder layer.

  This layer takes low- and high-level features as input and uses a dual-ASPP
  and dual-decoder structure to aggregate features for semantic and instance
  segmentation. On top of the decoders, three heads are used to predict semantic
  segmentation, instance center probabilities, and instance center regression
  per pixel.
  """

  def __init__(self,
               decoder_options,
               panoptic_deeplab_options,
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Initializes a Panoptic-DeepLab decoder.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      panoptic_deeplab_options: Model options as defined in
        config_pb2.ModelOptions.PanopticDeeplabOptions.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
    """
    super(PanopticDeepLab, self).__init__(name='PanopticDeepLab')

    low_level_feature_keys = [
        item.feature_key for item in panoptic_deeplab_options.low_level
    ]
    low_level_channels_project = [
        item.channels_project for item in panoptic_deeplab_options.low_level
    ]

    self._semantic_decoder = PanopticDeepLabSingleDecoder(
        high_level_feature_name=decoder_options.feature_key,
        low_level_feature_names=low_level_feature_keys,
        low_level_channels_project=low_level_channels_project,
        aspp_output_channels=decoder_options.aspp_channels,
        decoder_output_channels=decoder_options.decoder_channels,
        atrous_rates=decoder_options.atrous_rates,
        name='semantic_decoder',
        aspp_use_only_1x1_proj_conv=decoder_options.aspp_use_only_1x1_proj_conv,
        decoder_conv_type=decoder_options.decoder_conv_type,
        bn_layer=bn_layer,
        activation=activation)
    self._semantic_head = PanopticDeepLabSingleHead(
        panoptic_deeplab_options.semantic_head.head_channels,
        panoptic_deeplab_options.semantic_head.output_channels,
        common.PRED_SEMANTIC_LOGITS_KEY,
        name='semantic_head',
        conv_type=panoptic_deeplab_options.semantic_head.head_conv_type,
        bn_layer=bn_layer,
        activation=activation)

    self._instance_decoder = None
    self._instance_center_head = None
    self._instance_regression_head = None

    if panoptic_deeplab_options.instance.enable:
      if panoptic_deeplab_options.instance.low_level_override:
        low_level_options = panoptic_deeplab_options.instance.low_level_override
      else:
        low_level_options = panoptic_deeplab_options.low_level

      # If instance_decoder is set, use those options; otherwise reuse the
      # architecture as defined for the semantic decoder.
      if panoptic_deeplab_options.instance.HasField(
          'instance_decoder_override'):
        decoder_options = (panoptic_deeplab_options.instance
                           .instance_decoder_override)

      low_level_feature_keys = [item.feature_key for item in low_level_options]
      low_level_channels_project = [
          item.channels_project for item in low_level_options
      ]

      self._instance_decoder = PanopticDeepLabSingleDecoder(
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
          bn_layer=bn_layer,
          activation=activation)
      self._instance_center_head = PanopticDeepLabSingleHead(
          panoptic_deeplab_options.instance.center_head.head_channels,
          panoptic_deeplab_options.instance.center_head.output_channels,
          common.PRED_CENTER_HEATMAP_KEY,
          name='instance_center_head',
          conv_type=(
              panoptic_deeplab_options.instance.center_head.head_conv_type),
          bn_layer=bn_layer,
          activation=activation)
      self._instance_regression_head = PanopticDeepLabSingleHead(
          panoptic_deeplab_options.instance.regression_head.head_channels,
          panoptic_deeplab_options.instance.regression_head.output_channels,
          common.PRED_OFFSET_MAP_KEY,
          name='instance_regression_head',
          conv_type=(
              panoptic_deeplab_options.instance.regression_head.head_conv_type),
          bn_layer=bn_layer,
          activation=activation)

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

      if results.keys() & instance_center_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'center branch overlap. Please use unique keys.')
      results.update(instance_center_predictions)

      if results.keys() & instance_regression_predictions.keys():
        raise ValueError('The keys of the semantic branch and the instance '
                         'regression branch overlap. Please use unique keys.')
      results.update(instance_regression_predictions)

    return results
