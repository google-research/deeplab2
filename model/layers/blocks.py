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

"""Implements building blocks for neural networks."""
from typing import Optional

from absl import logging

import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import convolutions
from deeplab2.model.layers import squeeze_and_excite

backend = tf.keras.backend
layers = tf.keras.layers


class InvertedBottleneckBlock(tf.keras.layers.Layer):
  """An inverted bottleneck block.

  Reference:
  Sandler, M., Howard, A., et al. Mobilenetv2: Inverted residuals and linear
    bottlenecks. In CVPR, 2018
  Howard, A., Sandler, M., et al. Searching for mobilenetv3. In ICCV, 2019
  """

  def __init__(self,
               in_filters: int,
               out_filters: int,
               expand_ratio: int,
               strides: int,
               kernel_size: int = 3,
               se_ratio: Optional[float] = None,
               activation: str = 'relu',
               se_inner_activation: str = 'relu',
               se_gating_activation: str = 'sigmoid',
               depthwise_activation: Optional[str] = None,
               expand_se_in_filters: bool = False,
               atrous_rate: int = 1,
               divisible_by: int = 1,
               bn_layer: layers.Layer = tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay: float = 0.0,
               regularize_depthwise: bool = False,
               use_depthwise: bool = True,
               use_residual: bool = True,
               name: Optional[str] = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Initializes an inverted bottleneck block with BN after convolutions.

    Args:
      in_filters: The number of filters of the input tensor.
      out_filters: The number of filters of the output tensor.
      expand_ratio: The expand_ratio for an inverted bottleneck block. If
        expand_ratio is <= 1, this argument will be ignored.
      strides: The number of stride. If greater than 1, this block will
        ultimately downsample the input.
      kernel_size: The kernel size of the depthwise conv layer.
      se_ratio: If not None, se ratio for the squeeze and excitation layer.
      activation: The name of the activation function.
      se_inner_activation: The name of squeeze-excitation inner activation.
      se_gating_activation: The name of squeeze-excitation gating activation.
      depthwise_activation: The name of the activation function for depthwise
        only.
      expand_se_in_filters: Whether or not to expand in_filter in squeeze and
        excitation layer.
      atrous_rate: The atrous dilation rate to use for.
      divisible_by: A number that all inner dimensions are divisible by.
      bn_layer: An optional tf.keras.layers.Layer that computes the
          normalization (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: The weight decay for convolution kernels.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      use_depthwise: Whether to uses standard convolutions instead of depthwise.
      use_residual: Whether to include residual connection between input and
        output.
      name: Name for the block.
    """
    super(InvertedBottleneckBlock, self).__init__(name=name)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._expand_ratio = expand_ratio
    self._strides = strides
    self._kernel_size = kernel_size
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._atrous_rate = atrous_rate
    self._regularize_depthwise = regularize_depthwise
    self._use_depthwise = use_depthwise
    self._use_residual = use_residual
    self._activation = activation
    self._se_inner_activation = se_inner_activation
    self._se_gating_activation = se_gating_activation
    self._depthwise_activation = depthwise_activation
    self._expand_se_in_filters = expand_se_in_filters

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

    if depthwise_activation is None:
      self._depthwise_activation = activation

    if regularize_depthwise:
      depthwise_kernel_weight_decay = conv_kernel_weight_decay
    else:
      depthwise_kernel_weight_decay = 0.0

    if self._expand_ratio <= 1 and not self._use_depthwise:
      raise ValueError(
          'Undefined behavior if expand_ratio <= 1 and not use_depthwise')

    expand_filters = self._in_filters
    if self._expand_ratio > 1:
      # First 1x1 conv for channel expansion.
      expand_filters = utils.make_divisible(
          self._in_filters * self._expand_ratio, self._divisible_by)

      expand_kernel = 1 if self._use_depthwise else self._kernel_size
      expand_stride = 1 if self._use_depthwise else self._strides

      self._conv1_bn_act = convolutions.Conv2DSame(
          output_channels=expand_filters,
          kernel_size=expand_kernel,
          strides=expand_stride,
          atrous_rate=1,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation=self._activation,
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          name='expand_conv')

    if self._use_depthwise:
      # Depthwise conv.
      self._conv2_bn_act = convolutions.DepthwiseConv2DSame(
          kernel_size=self._kernel_size,
          strides=self._strides,
          atrous_rate=self._atrous_rate,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation=self._depthwise_activation,
          name='depthwise_conv')

    # Squeeze and excitation.
    if self._se_ratio is not None and self._se_ratio > 0:
      if self._expand_se_in_filters:
        in_filters = expand_filters
      else:
        in_filters = self._in_filters
      self._squeeze_excitation = squeeze_and_excite.SqueezeAndExcite(
          in_filters=in_filters,
          out_filters=expand_filters,
          se_ratio=self._se_ratio,
          divisible_by=self._divisible_by,
          kernel_initializer='he_normal',
          kernel_regularizer=tf.keras.regularizers.l2(conv_kernel_weight_decay),
          activation=self._se_inner_activation,
          gating_activation=self._se_gating_activation,
          name=name + '_se')
    else:
      logging.info(
          'Squeeze and Excitation is skipped due to undefined se_ratio')
      self._squeeze_excitation = None

    # Last 1x1 conv.
    self._conv3_bn = convolutions.Conv2DSame(
        output_channels=self._out_filters,
        kernel_size=1,
        strides=1,
        atrous_rate=1,
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=None,
        conv_kernel_weight_decay=conv_kernel_weight_decay,
        name='project_conv')

  def call(self, inputs, training=None):
    depthwise_output = None
    shortcut = inputs
    if self._expand_ratio > 1:
      x = self._conv1_bn_act(inputs, training=training)
    else:
      x = inputs

    if self._use_depthwise:
      x = self._conv2_bn_act(x, training=training)
      depthwise_output = x

    if self._squeeze_excitation is not None:
      x = self._squeeze_excitation(x)

    x = self._conv3_bn(x, training=training)

    if (self._use_residual and
        self._in_filters == self._out_filters):
      x = tf.add(x, shortcut)

    return x, depthwise_output
