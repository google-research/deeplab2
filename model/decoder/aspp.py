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

"""This file contains code to build an ASPP layer.

Reference:
  - [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
  - [ParseNet: Looking Wider to See Better](
      https://arxiv.org/pdf/1506.04579.pdf).
"""
from absl import logging
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import convolutions


layers = tf.keras.layers
backend = tf.keras.backend


class ASPPConv(tf.keras.layers.Layer):
  """An atrous convolution for ASPP."""

  def __init__(self,
               output_channels,
               atrous_rate,
               name,
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Creates a atrous convolution layer for the ASPP.

    This layer consists of an atrous convolution followed by a BatchNorm layer
    and a ReLU activation.

    Args:
      output_channels: An integer specifying the number of output channels of
        the convolution.
      atrous_rate: An integer specifying the atrous/dilation rate of the
        convolution.
      name: A string specifying the name of this layer.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
    """
    super(ASPPConv, self).__init__(name=name)

    self._conv_bn_act = convolutions.Conv2DSame(
        output_channels,
        kernel_size=3,
        name='conv_bn_act',
        atrous_rate=atrous_rate,
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    return self._conv_bn_act(input_tensor, training=training)


class ASPPPool(tf.keras.layers.Layer):
  """A pooling layer for ASPP."""

  def __init__(self,
               output_channels,
               name,
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Creates a pooling layer for the ASPP.

    This layer consists of a global average pooling, followed by a convolution,
    and by a BatchNorm layer and a ReLU activation.

    Args:
      output_channels: An integer specifying the number of output channels of
        the convolution.
      name: A string specifying the name of this layer.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
    """
    super(ASPPPool, self).__init__(name=name)

    self._pool_size = (None, None)
    self._conv_bn_act = convolutions.Conv2DSame(
        output_channels,
        kernel_size=1,
        name='conv_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation)

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the pooling layer.

    The default behavior of the pooling layer is global average pooling. A
    custom pooling size can be set here.

    Args:
      pool_size: A tuple specifying the pooling size of the pooling layer.

    Raises:
      An error occurs if exactly one pooling dimension is set to 'None'.
    """
    # If exactly one pooling dimension is 'None' raise an error.
    if None in pool_size and pool_size != (None, None):
      raise ValueError('The ASPP pooling layer requires that the pooling size '
                       'is set explicitly for both dimensions. In case, global '
                       'average pooling should be used, call '
                       'reset_pooling_layer() or set both to None.')

    self._pool_size = pool_size
    logging.info('Global average pooling in the ASPP pooling layer was replaced'
                 ' with tiled average pooling using the provided pool_size. '
                 'Please make sure this behavior is intended.')

  def get_pool_size(self):
    return self._pool_size

  def reset_pooling_layer(self):
    """Resets the pooling layer to global average pooling."""
    self._pool_size = (None, None)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if tuple(self._pool_size) == (None, None):
      # Global image pooling
      pool_size = input_tensor.shape[1:3]
    else:
      # Tiled image pooling
      pool_size = self._pool_size

    x = backend.pool2d(input_tensor, pool_size, padding='valid',
                       pool_mode='avg')
    x = self._conv_bn_act(x, training=training)

    target_h = tf.shape(input_tensor)[1]
    target_w = tf.shape(input_tensor)[2]

    x = utils.resize_align_corners(x, [target_h, target_w])
    return x


class ASPP(tf.keras.layers.Layer):
  """An atrous spatial pyramid pooling layer."""

  def __init__(self,
               output_channels,
               atrous_rates,
               aspp_use_only_1x1_proj_conv=False,
               name='ASPP',
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu'):
    """Creates an ASPP layer.

    Args:
      output_channels: An integer specifying the number of output channels of
        each ASPP convolution layer.
      atrous_rates: A list of three integers specifying the atrous/dilation rate
        of each ASPP convolution layer.
      aspp_use_only_1x1_proj_conv: Boolean, specifying if the ASPP five branches
        are turned off or not. If True, the ASPP module is degenerated to one
        1x1 convolution, projecting the input channels to `output_channels`.
      name: A string specifying the name of this layer (default: 'ASPP').
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.

    Raises:
      ValueError: An error occurs when atrous_rates does not contain 3
        elements and `aspp_use_only_1x1_proj_conv` is False.
    """
    super(ASPP, self).__init__(name=name)

    if not aspp_use_only_1x1_proj_conv and len(atrous_rates) != 3:
      raise ValueError(
          'The ASPP layers need exactly 3 atrous rates, but %d were given' %
          len(atrous_rates))
    self._aspp_use_only_1x1_proj_conv = aspp_use_only_1x1_proj_conv

    # Projection convolution is always used.
    self._proj_conv_bn_act = convolutions.Conv2DSame(
        output_channels,
        kernel_size=1,
        name='proj_conv_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation)

    if not aspp_use_only_1x1_proj_conv:
      self._conv_bn_act = convolutions.Conv2DSame(
          output_channels,
          kernel_size=1,
          name='conv_bn_act',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation=activation)
      rate1, rate2, rate3 = atrous_rates
      self._aspp_conv1 = ASPPConv(output_channels, rate1, name='aspp_conv1',
                                  bn_layer=bn_layer, activation=activation)
      self._aspp_conv2 = ASPPConv(output_channels, rate2, name='aspp_conv2',
                                  bn_layer=bn_layer, activation=activation)
      self._aspp_conv3 = ASPPConv(output_channels, rate3, name='aspp_conv3',
                                  bn_layer=bn_layer, activation=activation)
      self._aspp_pool = ASPPPool(output_channels, name='aspp_pool',
                                 bn_layer=bn_layer, activation=activation)
      # Dropout is needed only when ASPP five branches are used.
      self._proj_drop = layers.Dropout(rate=0.1)

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layer.

    The default behavior of the pooling layer is global average pooling. A
    custom pooling size can be set here.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    if not self._aspp_use_only_1x1_proj_conv:
      self._aspp_pool.set_pool_size(pool_size)

  def get_pool_size(self):
    if not self._aspp_use_only_1x1_proj_conv:
      return self._aspp_pool.get_pool_size()
    else:
      return (None, None)

  def reset_pooling_layer(self):
    """Resets the pooling layer to global average pooling."""
    self._aspp_pool.reset_pooling_layer()

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if self._aspp_use_only_1x1_proj_conv:
      x = self._proj_conv_bn_act(input_tensor, training=training)
    else:
      # Apply the ASPP module.
      results = []
      results.append(self._conv_bn_act(input_tensor, training=training))
      results.append(self._aspp_conv1(input_tensor, training=training))
      results.append(self._aspp_conv2(input_tensor, training=training))
      results.append(self._aspp_conv3(input_tensor, training=training))
      results.append(self._aspp_pool(input_tensor, training=training))
      x = tf.concat(results, 3)
      x = self._proj_conv_bn_act(x, training=training)
      x = self._proj_drop(x, training=training)
    return x
