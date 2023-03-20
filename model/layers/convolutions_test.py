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

"""Tests for convolutions."""

import numpy as np
import tensorflow as tf

from deeplab2.model.layers import convolutions
from deeplab2.utils import test_utils


class ConvolutionsTest(tf.test.TestCase):

  def test_conv2dsame_logging(self):
    with self.assertLogs(level='WARN'):
      _ = convolutions.Conv2DSame(
          output_channels=1,
          kernel_size=8,
          strides=2,
          name='conv',
          use_bn=False,
          activation=None)

  def test_conv2dsame_conv(self):
    conv = convolutions.Conv2DSame(
        output_channels=1,
        kernel_size=1,
        name='conv',
        use_bn=False,
        activation=None)
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 5))

    predicted_tensor = conv(input_tensor)
    expected_tensor = np.dot(input_tensor.numpy(),
                             conv._conv.get_weights()[0])[..., 0, 0]

    # Compare only up to 5 decimal digits to account for numerical accuracy.
    np.testing.assert_almost_equal(
        predicted_tensor.numpy(), expected_tensor, decimal=5)

  def test_conv2dsame_relu(self):
    conv = convolutions.Conv2DSame(
        output_channels=1,
        kernel_size=1,
        name='conv',
        activation='relu',
        use_bn=False)
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 5))

    predicted_tensor = conv(input_tensor)
    expected_tensor = np.dot(input_tensor.numpy(),
                             conv._conv.get_weights()[0])[..., 0, 0]
    expected_tensor[expected_tensor < 0.0] = 0.0

    # Compare only up to 5 decimal digits to account for numerical accuracy.
    np.testing.assert_almost_equal(
        predicted_tensor.numpy(), expected_tensor, decimal=5)

  def test_conv2dsame_relu6(self):
    conv = convolutions.Conv2DSame(
        output_channels=1,
        kernel_size=1,
        name='conv',
        activation='relu6',
        use_bn=False)
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 5)) * 10.

    predicted_tensor = conv(input_tensor)
    expected_tensor = np.dot(input_tensor.numpy(),
                             conv._conv.get_weights()[0])[..., 0, 0]
    expected_tensor[expected_tensor < 0.0] = 0.0
    expected_tensor[expected_tensor > 6.0] = 6.0

    # Compare only up to 5 decimal digits to account for numerical accuracy.
    np.testing.assert_almost_equal(
        predicted_tensor.numpy(), expected_tensor, decimal=5)

  def test_conv2dsame_shape(self):
    conv = convolutions.Conv2DSame(
        output_channels=64,
        kernel_size=7,
        strides=2,
        name='conv',
        use_bias=False,
        use_bn=True)
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 3))

    predicted_tensor = conv(input_tensor)
    expected_shape = [2, 90, 90, 64]

    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)

  @test_utils.test_all_strategies
  def test_conv2d_sync_bn(self, strategy):
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 3))

    for bn_layer in test_utils.NORMALIZATION_LAYERS:
      with strategy.scope():
        conv = convolutions.Conv2DSame(
            output_channels=64,
            kernel_size=7,
            strides=2,
            name='conv',
            use_bias=False,
            use_bn=True,
            bn_layer=bn_layer)
        conv(input_tensor)

  def test_depthwise_conv(self):
    conv = convolutions.DepthwiseConv2DSame(
        kernel_size=1, use_bn=False, use_bias=True, activation=None,
        name='conv')
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 5))

    predicted_tensor = conv(input_tensor)
    expected_tensor = (
        input_tensor.numpy() * conv._depthwise_conv.get_weights()[0][..., 0])

    np.testing.assert_equal(predicted_tensor.numpy(), expected_tensor)

  def test_depthwise_relu(self):
    conv = convolutions.DepthwiseConv2DSame(
        kernel_size=1, use_bn=False, activation='relu', name='conv')
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 5))

    predicted_tensor = conv(input_tensor)
    expected_tensor = (
        input_tensor.numpy() * conv._depthwise_conv.get_weights()[0][..., 0])
    expected_tensor[expected_tensor < 0.0] = 0.0

    np.testing.assert_equal(predicted_tensor.numpy(), expected_tensor)

  def test_depthwise_shape(self):
    conv = convolutions.DepthwiseConv2DSame(
        kernel_size=3, use_bn=True, use_bias=False, activation='relu',
        name='conv')
    input_shape = [2, 180, 180, 5]
    input_tensor = tf.random.uniform(shape=input_shape)

    predicted_tensor = conv(input_tensor)
    expected_shape = input_shape

    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)

  def test_depthwise_shape_with_stride2(self):
    conv = convolutions.DepthwiseConv2DSame(
        kernel_size=3, use_bn=True, use_bias=False, activation='relu',
        strides=2, name='conv')
    input_shape = [2, 181, 181, 5]
    input_tensor = tf.random.uniform(shape=input_shape)

    predicted_tensor = conv(input_tensor)
    expected_shape = [2, 91, 91, 5]

    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)

  @test_utils.test_all_strategies
  def test_depthwise_sync_bn(self, strategy):
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 3))

    for bn_layer in test_utils.NORMALIZATION_LAYERS:
      with strategy.scope():
        conv = convolutions.DepthwiseConv2DSame(
            kernel_size=7,
            name='conv',
            use_bn=True,
            use_bias=False,
            bn_layer=bn_layer,
            activation='relu')
        _ = conv(input_tensor)

  def test_global_context(self):
    input_tensor = tf.random.uniform(shape=(2, 180, 180, 3))
    global_context = convolutions.GlobalContext(name='global_context')
    output_tensor = global_context(input_tensor)
    # global_context is supposed to not change any values before training.
    np.testing.assert_array_almost_equal(input_tensor.numpy(),
                                         output_tensor.numpy())

  def test_switchable_atrous_conv_class(self):
    # Tests Switchable Atrous Convolution by equations.
    input_tensor = tf.random.uniform(shape=(3, 180, 180, 32))
    sac_layer = convolutions.SwitchableAtrousConvolution(
        64,
        kernel_size=3,
        padding='same',
        name='sac_conv')
    switch_conv = sac_layer._switch
    _ = switch_conv(input_tensor)
    switch_conv.kernel = tf.random.uniform(
        switch_conv.kernel.shape,
        minval=-1,
        maxval=1,
        dtype=switch_conv.kernel.dtype)
    switch_conv.bias = tf.random.uniform(
        switch_conv.bias.shape,
        minval=-1,
        maxval=1,
        dtype=switch_conv.bias.dtype)
    small_conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding='same',
        dilation_rate=1,
        name='small_conv')
    large_conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding='same',
        dilation_rate=3,
        name='large_conv')
    _ = small_conv(input_tensor)
    _ = large_conv(input_tensor)
    outputs = sac_layer(input_tensor)
    small_conv.kernel = sac_layer.kernel
    large_conv.kernel = sac_layer.kernel
    # Compute the expected outputs.
    switch_outputs = sac_layer._switch(sac_layer._average_pool(input_tensor))
    large_outputs = large_conv(input_tensor)
    small_outputs = small_conv(input_tensor)
    expected_outputs = (switch_outputs * large_outputs +
                        (1 - switch_outputs) * small_outputs)
    np.testing.assert_array_almost_equal(expected_outputs.numpy(),
                                         outputs.numpy())

  def test_switchable_atrous_conv_in_conv2dsame(self):
    # Tests Switchable Atrous Convolution in Conv2DSame.
    input_tensor = tf.random.uniform(shape=(3, 180, 180, 32))
    layer = convolutions.Conv2DSame(
        output_channels=64,
        kernel_size=7,
        strides=1,
        name='conv',
        use_bias=False,
        use_bn=True,
        use_switchable_atrous_conv=True,
        use_global_context_in_sac=True)
    output_tensor = layer(input_tensor)
    np.testing.assert_array_almost_equal(output_tensor.shape.as_list(),
                                         [3, 180, 180, 64])

  def test_conv1d_shape(self):
    conv = convolutions.Conv1D(
        output_channels=64,
        name='conv',
        use_bias=False,
        use_bn=True)
    input_tensor = tf.random.uniform(shape=(2, 180, 3))
    predicted_tensor = conv(input_tensor)
    expected_shape = [2, 180, 64]
    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)

  def test_separable_conv2d_same_output_shape(self):
    conv = convolutions.SeparableConv2DSame(
        output_channels=64,
        kernel_size=3,
        name='conv')
    input_tensor = tf.random.uniform(shape=(2, 5, 5, 3))
    predicted_tensor = conv(input_tensor)
    expected_shape = [2, 5, 5, 64]
    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)

  def test_stacked_conv2d_same_output_shape(self):
    conv = convolutions.StackedConv2DSame(
        num_layers=2,
        conv_type='depthwise_separable_conv',
        output_channels=64,
        kernel_size=3,
        name='conv')
    input_tensor = tf.random.uniform(shape=(2, 5, 5, 3))
    predicted_tensor = conv(input_tensor)
    expected_shape = [2, 5, 5, 64]
    self.assertListEqual(predicted_tensor.shape.as_list(), expected_shape)


if __name__ == '__main__':
  tf.test.main()
