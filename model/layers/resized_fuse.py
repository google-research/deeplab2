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

"""Implements a resized feature fuser for stacked decoders in MaX-DeepLab.

Reference:
  MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
    CVPR 2021. https://arxiv.org/abs/2012.00759
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""

import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import convolutions


class ResizedFuse(tf.keras.layers.Layer):
  """Fuses features by resizing and 1x1 convolutions.

  This function fuses all input features to a desired shape, by projecting the
  features to the desired number of channels, bilinear resizing the outputs
  (either upsampling or downsampling), and finally adding the outputs. If the
  input channel equals the desired output channels, the 1x1 convolutional
  projection is skipped. If the projection and bilinear resizing can be fused
  into a stride 2 convolution, we use this faster implementation. Other strides
  are also supported with the bilinear resizing, but are probably slower than
  strided convolutions.

  Reference:
    MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(self,
               name,
               height,
               width,
               num_channels,
               activation='relu',
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0):
    """Initializes a ResizedFuse layer.

    Args:
      name: A string, the name of this layer.
      height: An integer, the desired height of the output.
      width: An integer, the desired width of the output.
      num_channels: An integer, the num of output channels.
      activation: A string, type of activation function to apply.
      bn_layer: A tf.keras.layers.Layer that computes the normalization
        (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
    """
    super(ResizedFuse, self).__init__(name=name)
    self._height = height
    self._width = width
    self._num_channels = num_channels
    self._activation_fn = activations.get_activation(activation)
    self._bn_layer = bn_layer
    self._conv_kernel_weight_decay = conv_kernel_weight_decay

  def build(self, input_shapes):
    for index, feature_shape in enumerate(input_shapes):
      _, feature_height, feature_width, feature_channels = feature_shape
      if feature_channels == self._num_channels:
        continue
      elif ((feature_height + 1) // 2 == self._height and
            (feature_width + 1) // 2 == self._width):
        # Use stride 2 convolution to accelerate the operation if it generates
        # the desired spatial shape. Otherwise, the more general 1x1 convolution
        # and bilinear resizing are applied.

        # In a stacked decoder, we follow relu-conv-bn because we do the feature
        # summation before relu and after bn (following ResNet bottleneck
        # design). This ordering makes it easier to implement. Besides, it
        # avoids using many 1x1 convolutions when the input has a correct shape.
        current_name = '_strided_conv_bn{}'.format(index + 1)
        utils.safe_setattr(
            self, current_name, convolutions.Conv2DSame(
                self._num_channels, 1, current_name[1:],
                strides=2,
                use_bias=False,
                use_bn=True,
                bn_layer=self._bn_layer,
                activation='none',
                conv_kernel_weight_decay=self._conv_kernel_weight_decay))
      else:
        # If the input channel does not match that of the output, and the
        # operation cannot be accelerated by stride 2 convolution, then we
        # perform a flexible operation as follows. We first project the feature
        # to the desired number of channels, and then bilinearly resize the
        # output to the desired spatial resolution.
        current_name = '_resized_conv_bn{}'.format(index + 1)
        utils.safe_setattr(
            self, current_name, convolutions.Conv2DSame(
                self._num_channels, 1, current_name[1:],
                use_bias=False,
                use_bn=True,
                bn_layer=self._bn_layer,
                activation='none',
                conv_kernel_weight_decay=self._conv_kernel_weight_decay))

  def call(self, inputs, training=False):
    """Performs a forward pass.

    Args:
      inputs: A list of input [batch, input_height, input_width, input_channels]
        tensors to fuse, where each input tensor may have different spatial
        resolutions and number of channels.
      training: A boolean, whether the model is in training mode.

    Returns:
      output: A fused feature [batch, height, width, num_channels] tensor.
    """

    output_features = []
    for index, feature in enumerate(inputs):
      _, feature_height, feature_width, feature_channels = (
          feature.get_shape().as_list())
      if feature_channels == self._num_channels:
        # Resize the input feature if the number of channels equals the output.
        # We do not use a 1x1 convolution for this case because the previous
        # operation and the next operation are usually also 1x1 convolutions.
        # Besides, in stacked decoder, a feature can be reused many time, so it
        # saves parameter to avoid those many 1x1 convolutions.
        output_features.append(utils.resize_bilinear(
            feature, [self._height, self._width],
            align_corners=True))
      elif ((feature_height + 1) // 2 == self._height and
            (feature_width + 1) // 2 == self._width):
        current_name = '_strided_conv_bn{}'.format(index + 1)
        feature = self._activation_fn(feature)
        feature = getattr(self, current_name)(feature, training=training)
        output_features.append(feature)
      else:
        current_name = '_resized_conv_bn{}'.format(index + 1)
        feature = self._activation_fn(feature)
        feature = getattr(self, current_name)(feature, training=training)
        output_features.append(utils.resize_bilinear(
            feature, [self._height, self._width],
            align_corners=True))
      # Set the spatial shape of each output feature if possible.
      output_features[-1].set_shape(
          [None,
           self._height,
           self._width,
           self._num_channels])
    output = tf.add_n(output_features)
    return output
