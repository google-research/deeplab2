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

"""This file contains code to build MaX-DeepLab output heads.

Reference:
  MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
    CVPR 2021. https://arxiv.org/abs/2012.00759
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""
import math

import tensorflow as tf

from deeplab2 import common
from deeplab2.model.decoder import panoptic_deeplab
from deeplab2.model.layers import convolutions

_PIXEL_SPACE_FEATURE_KEY = 'pixel_space_feature'


def _get_transformer_class_head_num_classes(
    auxiliary_semantic_head_output_channels,
    ignore_label):
  """Computes the num of classes for the transformer class head.

  The transformer class head predicts non-void classes (i.e., thing classes and
  stuff classes) and a void (i.e., ∅, no object) class. If the auxiliary
  semantic head output channel includes the void class, e.g., on COCO, we
  directly use the semantic output channel. Otherwise, e.g., on Cityscapes, we
  add 1 (the void class) to the transformer class head.

  Args:
    auxiliary_semantic_head_output_channels: An integer, the number of output
      channels of the auxiliary semantic head (it should be the same as the
      num_classes field of the dataset information).
    ignore_label: An integer specifying the ignore label. Default to 255.

  Returns:
    num_classes: An integer, the num of classes for the transformer class head.
  """
  if ignore_label >= auxiliary_semantic_head_output_channels:
    return auxiliary_semantic_head_output_channels + 1
  else:
    return auxiliary_semantic_head_output_channels


def add_bias_towards_void(transformer_class_logits, void_prior_prob=0.9):
  """Adds init bias towards the void (no object) class to the class logits.

  We initialize the void class with a large probability, similar to Section 3.3
  of the Focal Loss paper.

  Reference:
    Focal Loss for Dense Object Detection, ICCV 2017.
      https://arxiv.org/abs/1708.02002
        Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár.

  Args:
    transformer_class_logits: A [batch, num_mask_slots, num_classes] tensor, the
      class logits predicted by the transformer. It concats (num_classes - 1)
      non-void classes, including both thing classes and stuff classes, and the
      void class (the last channel). If the dataset class IDs do not follow this
      order, MaX-DeepLab loss functions will handle the mapping and thus the
      architecture still supports any dataset.
    void_prior_prob: A float, the desired probability (after softmax) of the
      void class at initialization. Defaults to 0.9 as in MaX-DeepLab.

  Returns:
    updated_transformer_class_logits: A [batch, num_mask_slots, num_classes]

  Raises:
    ValueError: If the rank of transformer_class_logits is not 3.
  """
  class_logits_shape = transformer_class_logits.get_shape().as_list()
  if len(class_logits_shape) != 3:
    raise ValueError('Input transformer_class_logits should have rank 3.')

  init_bias = [0.0] * class_logits_shape[-1]
  init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))

  # Broadcasting the 1D init_bias to the 3D transformer_class_logits.
  return transformer_class_logits + tf.constant(init_bias, dtype=tf.float32)


def batch_norm_on_an_extra_axis(inputs, bn_layer):
  """Applies a batch norm layer on an extra axis.

  This batch norm will be used on the pixel space mask logits in MaX-DeepLab to
  avoid careful initialization of previous layers and careful scaling of the
  resulting outputs. In addition, applying batch norm on an extra axis does not
  introduce an extra gamma and beta for each mask slot. Instead, the current
  gamma and beta are shared for all mask slots and do not introduce biases on
  mask slots.

  Args:
    inputs: A [batch, height, width, num_mask_slots] tensor.
    bn_layer: A batch norm tf.keras.layers.Layer on the last axis.

  Returns:
    outputs: A [batch, height, width, num_mask_slots] tensor.
  """
  expanded_inputs = tf.expand_dims(inputs, axis=-1)
  outputs = bn_layer(expanded_inputs)
  return tf.squeeze(outputs, axis=-1)


class MaXDeepLab(tf.keras.layers.Layer):
  """A MaX-DeepLab head layer."""

  def __init__(self,
               decoder_options,
               max_deeplab_options,
               ignore_label,
               bn_layer=tf.keras.layers.BatchNormalization,
               use_auxiliary_semantic_decoder=True,
               use_auxiliary_semantic_head=True,
               activation='relu'):
    """Initializes a MaX-DeepLab head.

    Args:
      decoder_options: Decoder options as defined in config_pb2.DecoderOptions.
      max_deeplab_options: Model options as defined in
        config_pb2.ModelOptions.MaXDeepLabOptions.
      ignore_label: An integer specifying the ignore label.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      use_auxiliary_semantic_decoder: A boolean, whether to use an axuliary
        semantic decoder to upsample the semantic features.
      use_auxiliary_semantic_head: A boolean, whether to use an auxiliary
        semantic head to generate semantic predictions.
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
    """
    super(MaXDeepLab, self).__init__(name='MaXDeepLab')

    use_auxiliary_semantic_decoder = (
        use_auxiliary_semantic_decoder and use_auxiliary_semantic_head)
    if use_auxiliary_semantic_decoder:
      low_level_feature_keys = [
          item.feature_key for item in max_deeplab_options.auxiliary_low_level
      ]
      low_level_channels_project = [
          item.channels_project
          for item in max_deeplab_options.auxiliary_low_level
      ]

      self._auxiliary_semantic_decoder = (
          panoptic_deeplab.PanopticDeepLabSingleDecoder(
              high_level_feature_name=decoder_options.feature_key,
              low_level_feature_names=low_level_feature_keys,
              low_level_channels_project=low_level_channels_project,
              aspp_output_channels=decoder_options.aspp_channels,
              decoder_output_channels=decoder_options.decoder_channels,
              atrous_rates=decoder_options.atrous_rates,
              name='auxiliary_semantic_decoder',
              aspp_use_only_1x1_proj_conv=decoder_options
              .aspp_use_only_1x1_proj_conv,
              decoder_conv_type=decoder_options.decoder_conv_type,
              bn_layer=bn_layer,
              activation=activation))
    if use_auxiliary_semantic_head:
      self._auxiliary_semantic_head = (
          panoptic_deeplab.PanopticDeepLabSingleHead(
              max_deeplab_options.auxiliary_semantic_head.head_channels,
              max_deeplab_options.auxiliary_semantic_head.output_channels,
              common.PRED_SEMANTIC_LOGITS_KEY,
              name='auxiliary_semantic_head',
              conv_type=max_deeplab_options.auxiliary_semantic_head
              .head_conv_type,
              bn_layer=bn_layer,
              activation=activation))

    self._pixel_space_head = panoptic_deeplab.PanopticDeepLabSingleHead(
        max_deeplab_options.pixel_space_head.head_channels,
        max_deeplab_options.pixel_space_head.output_channels,
        _PIXEL_SPACE_FEATURE_KEY,
        name='pixel_space_head',
        conv_type=max_deeplab_options.pixel_space_head.head_conv_type,
        bn_layer=bn_layer,
        activation=activation)

    self._transformer_mask_head = convolutions.Conv1D(
        output_channels=max_deeplab_options.pixel_space_head.output_channels,
        name='transformer_mask_head',
        use_bias=False,
        # Use bn to avoid careful initialization.
        use_bn=True,
        bn_layer=bn_layer,
        bn_gamma_initializer='ones',
        activation=None,
        kernel_initializer='he_normal',
        kernel_size=1,
        padding='valid')
    # The transformer class head predicts non-void classes (i.e., thing classes
    # and stuff classes) and a void (i.e., ∅, no object) class.
    num_classes = _get_transformer_class_head_num_classes(
        max_deeplab_options.auxiliary_semantic_head.output_channels,
        ignore_label=ignore_label)
    self._transformer_class_head = convolutions.Conv1D(
        output_channels=num_classes,
        name='transformer_class_head',
        # Use conv bias rather than bn on this final class logit output.
        use_bias=True,
        use_bn=False,
        activation=None,
        # Follow common ImageNet class initlization with stddev 0.01.
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        kernel_size=1,
        padding='valid')

    self._pixel_space_feature_batch_norm = bn_layer(
        axis=-1, name='pixel_space_feature_batch_norm',
        gamma_initializer=tf.keras.initializers.Constant(1.0))
    # Use a batch norm to avoid care initialization of the mask outputs.
    self._pixel_space_mask_batch_norm = bn_layer(
        axis=-1, name='pixel_space_mask_batch_norm',
        # Initialize the pixel space mask with a low temperature.
        gamma_initializer=tf.keras.initializers.Constant(0.1))

    self._use_auxiliary_semantic_decoder = use_auxiliary_semantic_decoder
    self._use_auxiliary_semantic_head = use_auxiliary_semantic_head

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layers to global average pooling."""
    if self._use_auxiliary_semantic_decoder:
      self._auxiliary_semantic_decoder.reset_pooling_layer()
    else:
      self._pool_size = (None, None)

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layers.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layers.
    """
    if self._use_auxiliary_semantic_decoder:
      self._auxiliary_semantic_decoder.set_pool_size(pool_size)
    else:
      self._pool_size = pool_size

  def get_pool_size(self):
    if self._use_auxiliary_semantic_decoder:
      return self._auxiliary_semantic_decoder.get_pool_size()
    else:
      return self._pool_size

  @property
  def checkpoint_items(self):
    items = {
        common.CKPT_PIXEL_SPACE_HEAD:
            self._pixel_space_head,
        common.CKPT_TRANSFORMER_MASK_HEAD:
            self._transformer_mask_head,
        common.CKPT_TRANSFORMER_CLASS_HEAD:
            self._transformer_class_head,
        common.CKPT_PIXEL_SPACE_FEATURE_BATCH_NORM:
            self._pixel_space_feature_batch_norm,
        common.CKPT_PIXEL_SPACE_MASK_BATCH_NORM:
            self._pixel_space_mask_batch_norm,
    }
    if self._use_auxiliary_semantic_decoder:
      items.update({
          common.CKPT_SEMANTIC_DECODER:
              self._auxiliary_semantic_decoder,
      })
    if self._use_auxiliary_semantic_head:
      items.update({
          common.CKPT_SEMANTIC_HEAD_WITHOUT_LAST_LAYER:
              self._auxiliary_semantic_head.conv_block,
          common.CKPT_SEMANTIC_LAST_LAYER:
              self._auxiliary_semantic_head.final_conv,
      })
    return items

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels] or [batch, length, channels]. Different keys should point to
        different features extracted by the encoder, e.g., low-level or
        high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the auxiliary semantic segmentation logits, the
        pixel space normalized feature, the pixel space mask logits, and the
        mask transformer class logits.
    """
    results = {}
    semantic_features = features['feature_semantic']
    panoptic_features = features['feature_panoptic']
    transformer_class_feature = features['transformer_class_feature']
    transformer_mask_feature = features['transformer_mask_feature']

    if self._use_auxiliary_semantic_head:
      # Auxiliary semantic head.
      semantic_shape = semantic_features.get_shape().as_list()
      panoptic_shape = panoptic_features.get_shape().as_list()
      # MaX-DeepLab always predicts panoptic feature at high resolution (e.g.,
      # stride 4 or stride 2), but the auxiliary semantic feature could be at
      # low resolution (e.g., stride 16 or stride 32), in the absence of the
      # stacked decoder (L == 0). In this case, we use an auxiliary semantic
      # decoder on top of the semantic feature, in order to add the auxiliary
      # semantic loss.
      if (semantic_shape[1:3] != panoptic_shape[1:3] and
          self._use_auxiliary_semantic_decoder):
        semantic_features = self._auxiliary_semantic_decoder(
            features, training=training)
      auxiliary_semantic_results = self._auxiliary_semantic_head(
          semantic_features, training=training)
      results.update(auxiliary_semantic_results)

    # Pixel space head.
    pixel_space_feature = self._pixel_space_head(
        panoptic_features, training=training)[_PIXEL_SPACE_FEATURE_KEY]
    pixel_space_feature = self._pixel_space_feature_batch_norm(
        pixel_space_feature)
    pixel_space_normalized_feature = tf.math.l2_normalize(
        pixel_space_feature, axis=-1)
    results[common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY] = (
        pixel_space_normalized_feature)

    # Transformer class head.
    transformer_class_logits = self._transformer_class_head(
        transformer_class_feature)
    # Bias towards the void class at initialization.
    transformer_class_logits = add_bias_towards_void(
        transformer_class_logits)
    results[common.PRED_TRANSFORMER_CLASS_LOGITS_KEY] = transformer_class_logits

    # Transformer mask kernel.
    transformer_mask_kernel = self._transformer_mask_head(
        transformer_mask_feature)

    # Convolutional mask head. The pixel space mask logits are the matrix
    # multiplication (or convolution) of the pixel space normalized feature and
    # the transformer mask kernel.
    pixel_space_mask_logits = tf.einsum(
        'bhwd,bid->bhwi',
        pixel_space_normalized_feature,
        transformer_mask_kernel)
    # The above multiplication constructs a second-order operation which is
    # sensitive to the feature scales and initializations. In order to avoid
    # careful initialization or scaling of the layers, we apply batch norms on
    # top of pixel_space_feature, transformer_mask_kernel, and the resulting
    # pixel_space_mask_logits.
    pixel_space_mask_logits = batch_norm_on_an_extra_axis(
        pixel_space_mask_logits, self._pixel_space_mask_batch_norm)
    results[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY] = (
        pixel_space_mask_logits)

    if common.PRED_AUXILIARY_OUTPUTS in features:
      results[common.PRED_AUXILIARY_OUTPUTS] = (
          features[common.PRED_AUXILIARY_OUTPUTS])

    return results
