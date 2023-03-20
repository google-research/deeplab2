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

"""Squeeze and excite layer.

This script implements the squeeze-and-excite (SE), proposed in
- Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie,
Gang Sun, Enhua Wu. In CVPR 2018.

Recently, this SE operation is further simplied with a single fully
connected layer, referred as simplified_squeeze_and_excite in our
implementation. For details, please see
- Lee and Park proposed to use only one fully connected layer in SE.
CenterMask : Real-Time Anchor-Free Instance Segmentation.
Youngwan Lee and Jongyoul Park. In CVPR 2020.
"""
from typing import Optional

from absl import logging
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations

layers = tf.keras.layers


class SimplifiedSqueezeAndExcite(tf.keras.layers.Layer):
  """A simplified squeeze-and-excite layer.

  Original squeeze-and-exciation (SE) is proposed in
  Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie,
  Gang Sun, Enhua Wu. In CVPR 2018.

  Lee and Park proposed to use only one fully connected layer in SE.
  CenterMask : Real-Time Anchor-Free Instance Segmentation.
  Youngwan Lee and Jongyoul Park. In CVPR 2020.

  In this function, we implement the simplified version of SE.

  Additionally, we follow MobileNetv3 to use the hard sigmoid function.
  """

  def __init__(self, squeeze_channels, name=None):
    """Initializes a simplified squeeze-and-excite layer.

    Args:
      squeeze_channels: Integer, channels for the squeezed features.
      name: An optional string specifying the operation name.
    """
    super(SimplifiedSqueezeAndExcite, self).__init__(name=name)
    self._squeeze_channels = squeeze_channels

    self._se_conv = layers.Conv2D(self._squeeze_channels,
                                  1,
                                  name='squeeze_and_excite',
                                  use_bias=True,
                                  kernel_initializer='VarianceScaling')
    self._hard_sigmoid = activations.get_activation('hard_sigmoid')

  def call(self, input_tensor):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].

    Returns:
      The output tensor.
    """
    pooled = tf.reduce_mean(input_tensor, [1, 2], keepdims=True)
    squeezed = self._se_conv(pooled)
    excited = self._hard_sigmoid(squeezed) * input_tensor
    return excited

  def get_config(self):
    config = {
        'squeeze_channels': self._squeeze_channels,
    }
    base_config = super(SimplifiedSqueezeAndExcite, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SqueezeAndExcite(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer.

  Reference: Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie,
  Gang Sun, Enhua Wu. In CVPR 2018.
  This implementation follows the original SE and differs from the above
  simplified version.
  """

  def __init__(
      self,
      in_filters: int,
      out_filters: int,
      se_ratio: float,
      divisible_by: int = 1,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activation: str = 'relu',
      gating_activation: str = 'sigmoid',
      name: Optional[str] = None):
    """Initializes a squeeze and excitation layer.

    Args:
      in_filters: The number of filters that se_ratio should be applied to.
      out_filters: The number of filters of the output tensor.
      se_ratio: The SE ratio for the squeeze and excitation layer.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      kernel_initializer: The kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: The name of the activation function.
      gating_activation: The name of the activation function for final
        gating function.
      name: The layer name.
    """
    super(SqueezeAndExcite, self).__init__(name=name)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._spatial_axis = [1, 2]
    else:
      self._spatial_axis = [2, 3]
    self._activation_fn = activations.get_activation(activation)
    self._gating_activation_fn = activations.get_activation(gating_activation)

    num_reduced_filters = utils.make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by)
    if self._se_ratio > 1.0:
      logging.warn('Squeezing ratio %d is larger than 1.0.', self._se_ratio)

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name=name + '_reduce')

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name=name + '_expand')

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._activation_fn(self._se_reduce(x))
    x = self._gating_activation_fn(self._se_expand(x))
    return x * inputs
