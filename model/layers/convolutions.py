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

"""This file contains wrapper classes for convolution layers of tf.keras and Switchable Atrous Convolution.

Switchable Atrous Convolution (SAC) is convolution with a switchable atrous
rate. It also has optional pre- and post-global context layers.
[1] Siyuan Qiao, Liang-Chieh Chen, Alan Yuille. DetectoRS: Detecting Objects
    with Recursive Feature Pyramid and Switchable Atrous Convolution.
    arXiv:2006.02334
"""
from typing import Optional

from absl import logging
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations


def _compute_padding_size(kernel_size, atrous_rate):
  kernel_size_effective = kernel_size + (kernel_size - 1) * (atrous_rate - 1)
  pad_total = kernel_size_effective - 1
  pad_begin = pad_total // 2
  pad_end = pad_total - pad_begin
  if pad_begin != pad_end:
    logging.warn('Convolution requires one more padding to the '
                 'bottom-right pixel. This may cause misalignment.')
  return (pad_begin, pad_end)


class GlobalContext(tf.keras.layers.Layer):
  """Class for the global context modules in Switchable Atrous Convolution."""

  def build(self, input_shape):
    super().build(input_shape)
    input_shape = tf.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
    self.convolution = tf.keras.layers.Conv2D(
        input_channel, 1, strides=1, padding='same', name=self.name + '_conv',
        kernel_initializer='zeros', bias_initializer='zeros')

  def call(self, inputs, *args, **kwargs):
    outputs = self.global_average_pooling(inputs)
    outputs = tf.expand_dims(outputs, axis=1)
    outputs = tf.expand_dims(outputs, axis=1)
    outputs = self.convolution(outputs)
    return inputs + outputs

  def _get_input_channel(self, input_shape):
    # Reference: tf.keras.layers.convolutional.Conv.
    if input_shape.dims[-1].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[-1])


class SwitchableAtrousConvolution(tf.keras.layers.Conv2D):
  """Class for the Switchable Atrous Convolution."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._average_pool = tf.keras.layers.AveragePooling2D(
        pool_size=(5, 5), strides=1, padding='same')
    self._switch = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        strides=self.strides,
        padding='same',
        dilation_rate=1,
        name='switch',
        kernel_initializer='zeros',
        bias_initializer='zeros')

  def large_convolution_op(self, inputs, kernel):
    if self.padding == 'causal':
      tf_padding = 'VALID'
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    large_dilation_rate = list(self.dilation_rate)
    large_dilation_rate = [r * 3 for r in large_dilation_rate]
    return tf.nn.convolution(
        inputs, kernel,
        strides=list(self.strides),
        padding=tf_padding,
        dilations=large_dilation_rate,
        data_format=self._tf_data_format,
        name=self.__class__.__name__ + '_large'
        )

  def call(self, inputs):
    # Reference: tf.keras.layers.convolutional.Conv.
    input_shape = inputs.shape
    switches = self._switch(self._average_pool(inputs))

    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = tf.compat.v1.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self.convolution_op(inputs, self.kernel)
    outputs_large = self.large_convolution_op(inputs, self.kernel)

    outputs = switches * outputs_large + (1 - switches) * outputs

    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs, self.bias, data_format=self._tf_data_format)

    if not tf.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def squeeze_batch_dims(self, inp, op, inner_rank):
    # Reference: tf.keras.utils.conv_utils.squeeze_batch_dims.
    with tf.name_scope('squeeze_batch_dims'):
      shape = inp.shape

      inner_shape = shape[-inner_rank:]
      if not inner_shape.is_fully_defined():
        inner_shape = tf.compat.v1.shape(inp)[-inner_rank:]

      batch_shape = shape[:-inner_rank]
      if not batch_shape.is_fully_defined():
        batch_shape = tf.compat.v1.shape(inp)[:-inner_rank]

      if isinstance(inner_shape, tf.TensorShape):
        inp_reshaped = tf.reshape(inp, [-1] + inner_shape.as_list())
      else:
        inp_reshaped = tf.reshape(
            inp, tf.concat(([-1], inner_shape), axis=-1))

      out_reshaped = op(inp_reshaped)

      out_inner_shape = out_reshaped.shape[-inner_rank:]
      if not out_inner_shape.is_fully_defined():
        out_inner_shape = tf.compat.v1.shape(out_reshaped)[-inner_rank:]

      out = tf.reshape(
          out_reshaped, tf.concat((batch_shape, out_inner_shape), axis=-1))

      out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
      return out


class Conv2DSame(tf.keras.layers.Layer):
  """A wrapper class for a 2D convolution with 'same' padding.

  In contrast to tf.keras.layers.Conv2D, this layer aligns the kernel with the
  top-left corner rather than the bottom-right corner. Optionally, a batch
  normalization and an activation can be added on top.
  """

  def __init__(
      self,
      output_channels: int,
      kernel_size: int,
      name: str,
      strides: int = 1,
      atrous_rate: int = 1,
      use_bias: bool = True,
      use_bn: bool = False,
      bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
      bn_gamma_initializer: str = 'ones',
      activation: Optional[str] = None,
      use_switchable_atrous_conv: bool = False,
      use_global_context_in_sac: bool = False,
      conv_kernel_weight_decay: float = 0.0):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Initializes convolution with zero padding aligned to the top-left corner.

    DeepLab aligns zero padding differently to tf.keras 'same' padding.
    Considering a convolution with a 7x7 kernel, a stride of 2 and an even input
    size, tf.keras 'same' padding will add 2 zero padding to the top-left and 3
    zero padding to the bottom-right. However, for consistent feature alignment,
    DeepLab requires an equal padding of 3 in all directions. This behavior is
    consistent with e.g. the ResNet 'stem' block.

    Args:
      output_channels: An integer specifying the number of filters of the
        convolution.
      kernel_size: An integer specifying the size of the convolution kernel.
      name: A string specifying the name of this layer.
      strides: An optional integer or tuple of integers specifying the size of
        the strides (default: 1).
      atrous_rate: An optional integer or tuple of integers specifying the
        atrous rate of the convolution (default: 1).
      use_bias: An optional flag specifying whether bias should be added for the
        convolution.
      use_bn: An optional flag specifying whether batch normalization should be
        added after the convolution (default: False).
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      bn_gamma_initializer: An initializer for the batch norm gamma weight.
      activation: An optional flag specifying an activation function to be added
        after the convolution.
      use_switchable_atrous_conv: Boolean, whether the layer uses switchable
        atrous convolution.
      use_global_context_in_sac: Boolean, whether the switchable atrous
        convolution (SAC) uses pre- and post-global context.
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.

    Raises:
      ValueError: If use_bias and use_bn in the convolution.
    """
    super(Conv2DSame, self).__init__(name=name)

    if use_bn and use_bias:
      raise ValueError('Conv2DSame is using convolution bias with batch_norm.')

    if use_global_context_in_sac:
      self._pre_global_context = GlobalContext(name='pre_global_context')

    convolution_op = tf.keras.layers.Conv2D
    convolution_padding = 'same'
    if strides == 1 or strides == (1, 1):
      if use_switchable_atrous_conv:
        convolution_op = SwitchableAtrousConvolution
    else:
      padding = _compute_padding_size(kernel_size, atrous_rate)
      self._zeropad = tf.keras.layers.ZeroPadding2D(
          padding=(padding, padding), name='zeropad')
      convolution_padding = 'valid'
    self._conv = convolution_op(
        output_channels,
        kernel_size,
        strides=strides,
        padding=convolution_padding,
        use_bias=use_bias,
        dilation_rate=atrous_rate,
        name='conv',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(
            conv_kernel_weight_decay))

    if use_global_context_in_sac:
      self._post_global_context = GlobalContext(name='post_global_context')

    if use_bn:
      self._batch_norm = bn_layer(axis=3, name='batch_norm',
                                  gamma_initializer=bn_gamma_initializer)

    self._activation_fn = None
    if activation is not None:
      self._activation_fn = activations.get_activation(activation)

    self._use_global_context_in_sac = use_global_context_in_sac
    self._strides = strides
    self._use_bn = use_bn

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
    x = input_tensor
    if self._use_global_context_in_sac:
      x = self._pre_global_context(x)

    if not (self._strides == 1 or self._strides == (1, 1)):
      x = self._zeropad(x)
    x = self._conv(x)

    if self._use_global_context_in_sac:
      x = self._post_global_context(x)

    if self._use_bn:
      x = self._batch_norm(x, training=training)

    if self._activation_fn is not None:
      x = self._activation_fn(x)
    return x


class DepthwiseConv2DSame(tf.keras.layers.Layer):
  """A wrapper class for a 2D depthwise convolution.

  In contrast to convolutions in tf.keras.layers.DepthwiseConv2D, this layers
  aligns the kernel with the top-left corner rather than the bottom-right
  corner. Optionally, a batch normalization and an activation can be added.
  """

  def __init__(self,
               kernel_size: int,
               name: str,
               strides: int = 1,
               atrous_rate: int = 1,
               use_bias: bool = True,
               use_bn: bool = False,
               bn_layer=tf.keras.layers.BatchNormalization,
               activation: Optional[str] = None):
    """Initializes a 2D depthwise convolution.

    Args:
      kernel_size: An integer specifying the size of the convolution kernel.
      name: A string specifying the name of this layer.
      strides: An optional integer or tuple of integers specifying the size of
        the strides (default: 1).
      atrous_rate: An optional integer or tuple of integers specifying the
        atrous rate of the convolution (default: 1).
      use_bias: An optional flag specifying whether bias should be added for the
        convolution.
      use_bn: An optional flag specifying whether batch normalization should be
        added after the convolution (default: False).
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: An optional flag specifying an activation function to be added
        after the convolution.

    Raises:
      ValueError: If use_bias and use_bn in the convolution.
    """
    super(DepthwiseConv2DSame, self).__init__(name=name)

    if use_bn and use_bias:
      raise ValueError(
          'DepthwiseConv2DSame is using convlution bias with batch_norm.')

    if strides == 1 or strides == (1, 1):
      convolution_padding = 'same'
    else:
      padding = _compute_padding_size(kernel_size, atrous_rate)
      self._zeropad = tf.keras.layers.ZeroPadding2D(
          padding=(padding, padding), name='zeropad')
      convolution_padding = 'valid'
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=convolution_padding,
        use_bias=use_bias,
        dilation_rate=atrous_rate,
        name='depthwise_conv')
    if use_bn:
      self._batch_norm = bn_layer(axis=3, name='batch_norm')

    self._activation_fn = None
    if activation is not None:
      self._activation_fn = activations.get_activation(activation)

    self._strides = strides
    self._use_bn = use_bn

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
    x = input_tensor
    if not (self._strides == 1 or self._strides == (1, 1)):
      x = self._zeropad(x)
    x = self._depthwise_conv(x)
    if self._use_bn:
      x = self._batch_norm(x, training=training)
    if self._activation_fn is not None:
      x = self._activation_fn(x)
    return x


class SeparableConv2DSame(tf.keras.layers.Layer):
  """A wrapper class for a 2D separable convolution.

  In contrast to convolutions in tf.keras.layers.SeparableConv2D, this layers
  aligns the kernel with the top-left corner rather than the bottom-right
  corner. Optionally, a batch normalization and an activation can be added.
  """

  def __init__(
      self,
      output_channels: int,
      kernel_size: int,
      name: str,
      strides: int = 1,
      atrous_rate: int = 1,
      use_bias: bool = True,
      use_bn: bool = False,
      bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
      activation: Optional[str] = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Initializes a 2D separable convolution.

    Args:
      output_channels: An integer specifying the number of filters of the
        convolution output.
      kernel_size: An integer specifying the size of the convolution kernel.
      name: A string specifying the name of this layer.
      strides: An optional integer or tuple of integers specifying the size of
        the strides (default: 1).
      atrous_rate: An optional integer or tuple of integers specifying the
        atrous rate of the convolution (default: 1).
      use_bias: An optional flag specifying whether bias should be added for the
        convolution.
      use_bn: An optional flag specifying whether batch normalization should be
        added after the convolution (default: False).
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: An optional flag specifying an activation function to be added
        after the convolution.

    Raises:
      ValueError: If use_bias and use_bn in the convolution.
    """
    super(SeparableConv2DSame, self).__init__(name=name)
    if use_bn and use_bias:
      raise ValueError(
          'SeparableConv2DSame is using convolution bias with batch_norm.')

    self._depthwise = DepthwiseConv2DSame(
        kernel_size=kernel_size,
        name='depthwise',
        strides=strides,
        atrous_rate=atrous_rate,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_layer=bn_layer,
        activation=activation)
    self._pointwise = Conv2DSame(
        output_channels=output_channels,
        kernel_size=1,
        name='pointwise',
        strides=1,
        atrous_rate=1,
        use_bias=use_bias,
        use_bn=use_bn,
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
    x = self._depthwise(input_tensor, training=training)
    return self._pointwise(x, training=training)


class StackedConv2DSame(tf.keras.layers.Layer):
  """Stacked Conv2DSame or SeparableConv2DSame.

  This class sequentially stacks a given number of Conv2DSame layers or
  SeparableConv2DSame layers.
  """

  def __init__(
      self,
      num_layers: int,
      conv_type: str,
      output_channels: int,
      kernel_size: int,
      name: str,
      strides: int = 1,
      atrous_rate: int = 1,
      use_bias: bool = True,
      use_bn: bool = False,
      bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
      activation: Optional[str] = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Initializes a stack of convolutions.

    Args:
      num_layers: The number of convolutions to create.
      conv_type: A string specifying the convolution type used in each block.
        Must be one of 'standard_conv' or 'depthwise_separable_conv'.
      output_channels: An integer specifying the number of filters of the
        convolution output.
      kernel_size: An integer specifying the size of the convolution kernel.
      name: A string specifying the name of this layer.
      strides: An optional integer or tuple of integers specifying the size of
        the strides (default: 1).
      atrous_rate: An optional integer or tuple of integers specifying the
        atrous rate of the convolution (default: 1).
      use_bias: An optional flag specifying whether bias should be added for the
        convolution.
      use_bn: An optional flag specifying whether batch normalization should be
        added after the convolution (default: False).
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      activation: An optional flag specifying an activation function to be added
        after the convolution.

    Raises:
      ValueError: An error occurs when conv_type is neither 'standard_conv'
        nor 'depthwise_separable_conv'.
    """
    super(StackedConv2DSame, self).__init__(name=name)
    if conv_type == 'standard_conv':
      convolution_op = Conv2DSame
    elif conv_type == 'depthwise_separable_conv':
      convolution_op = SeparableConv2DSame
    else:
      raise ValueError('Convolution %s not supported.' % conv_type)

    for index in range(num_layers):
      current_name = utils.get_conv_bn_act_current_name(index, use_bn,
                                                        activation)
      utils.safe_setattr(self, current_name, convolution_op(
          output_channels=output_channels,
          kernel_size=kernel_size,
          name=utils.get_layer_name(current_name),
          strides=strides,
          atrous_rate=atrous_rate,
          use_bias=use_bias,
          use_bn=use_bn,
          bn_layer=bn_layer,
          activation=activation))
    self._num_layers = num_layers
    self._use_bn = use_bn
    self._activation = activation

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
    x = input_tensor
    for index in range(self._num_layers):
      current_name = utils.get_conv_bn_act_current_name(index, self._use_bn,
                                                        self._activation)
      x = getattr(self, current_name)(x, training=training)
    return x


class Conv1D(tf.keras.layers.Layer):
  """A wrapper class for a 1D convolution with batch norm and activation.

  Conv1D creates a convolution kernel that is convolved with the layer input
  over a single spatial (or temporal) dimension to produce a tensor of outputs.
  The input should always be 3D with shape [batch, length, channel], so
  accordingly, the optional batch norm is done on axis=2.

  In DeepLab, we use Conv1D only with kernel_size = 1 for dual path transformer
  layers in MaX-DeepLab [1] architectures.

  Reference:
  [1] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
      CVPR 2021.
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(
      self,
      output_channels: int,
      name: str,
      use_bias: bool = True,
      use_bn: bool = False,
      bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
      bn_gamma_initializer: str = 'ones',
      activation: Optional[str] = None,
      conv_kernel_weight_decay: float = 0.0,
      kernel_initializer='he_normal',
      kernel_size: int = 1,
      padding: str = 'valid'):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Initializes a Conv1D.

    Args:
      output_channels: An integer specifying the number of filters of the
        convolution.
      name: A string specifying the name of this layer.
      use_bias: An optional flag specifying whether bias should be added for the
        convolution.
      use_bn: An optional flag specifying whether batch normalization should be
        added after the convolution (default: False).
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      bn_gamma_initializer: An initializer for the batch norm gamma weight.
      activation: An optional flag specifying an activation function to be added
        after the convolution.
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      kernel_initializer: An initializer for the convolution kernel.
      kernel_size: An integer specifying the size of the convolution kernel.
      padding: An optional string specifying the padding to use. Must be either
        'same' or 'valid' (default: 'valid').

    Raises:
      ValueError: If use_bias and use_bn in the convolution.
    """
    super(Conv1D, self).__init__(name=name)

    if use_bn and use_bias:
      raise ValueError('Conv1D is using convlution bias with batch_norm.')

    self._conv = tf.keras.layers.Conv1D(
        output_channels,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=use_bias,
        name='conv',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(
            conv_kernel_weight_decay))

    self._batch_norm = None
    if use_bn:
      # Batch norm uses axis=2 because the input is 3D with channel being the
      # last dimension.
      self._batch_norm = bn_layer(axis=2, name='batch_norm',
                                  gamma_initializer=bn_gamma_initializer)

    self._activation_fn = None
    if activation is not None:
      self._activation_fn = activations.get_activation(activation)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, length,
        channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    x = self._conv(input_tensor)
    if self._batch_norm is not None:
      x = self._batch_norm(x, training=training)
    if self._activation_fn is not None:
      x = self._activation_fn(x)
    return x
