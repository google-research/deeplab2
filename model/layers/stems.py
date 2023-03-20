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

"""This script contains STEMs for neural networks.

The `STEM` is defined as the first few convolutions that process the input
image to a spatially smaller feature map (e.g., output stride = 2).


Reference code:
https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py
"""
import tensorflow as tf

from deeplab2.model.layers import convolutions

layers = tf.keras.layers


class InceptionSTEM(tf.keras.layers.Layer):
  """A InceptionSTEM layer.

  This class builds an InceptionSTEM layer which can be used to as the first
  few layers in a neural network. In particular, InceptionSTEM contains three
  consecutive 3x3 colutions.

  Reference:
  - Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander Alemi.
    "Inception-v4, inception-resnet and the impact of residual connections on
    learning." In AAAI, 2017.
  """

  def __init__(self,
               bn_layer=tf.keras.layers.BatchNormalization,
               width_multiplier=1.0,
               conv_kernel_weight_decay=0.0,
               activation='relu'):
    """Creates the InceptionSTEM layer.

    Args:
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      width_multiplier: A float multiplier, controlling the value of
        convolution output channels.
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      activation: A string specifying an activation function to be used in this
        stem.
    """
    super(InceptionSTEM, self).__init__(name='stem')

    self._conv1_bn_act = convolutions.Conv2DSame(
        output_channels=int(64 * width_multiplier),
        kernel_size=3,
        name='conv1_bn_act',
        strides=2,
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation,
        conv_kernel_weight_decay=conv_kernel_weight_decay)

    self._conv2_bn_act = convolutions.Conv2DSame(
        output_channels=int(64 * width_multiplier),
        kernel_size=3,
        name='conv2_bn_act',
        strides=1,
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation,
        conv_kernel_weight_decay=conv_kernel_weight_decay)

    self._conv3_bn = convolutions.Conv2DSame(
        output_channels=int(128 * width_multiplier),
        kernel_size=3,
        strides=1,
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation='none',
        name='conv3_bn',
        conv_kernel_weight_decay=conv_kernel_weight_decay)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      Two output tensors. The first output tensor is not activated. The second
        tensor is activated.
    """
    x = self._conv1_bn_act(input_tensor, training=training)
    x = self._conv2_bn_act(x, training=training)
    x = self._conv3_bn(x, training=training)
    return x
