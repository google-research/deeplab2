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

"""Tests for mobilenet."""

from absl.testing import parameterized

import tensorflow as tf

from deeplab2.model import test_utils
from deeplab2.model.encoder import mobilenet


class MobilenetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters('MobileNetV3Small', 'MobileNetV3Large')
  def test_mobilenetv3_construct_graph(self, model_name):
    tf.keras.backend.set_image_data_format('channels_last')
    input_size = 128

    mobilenet_models = {
        'MobileNetV3Small': mobilenet.MobileNetV3Small,
        'MobileNetV3Large': mobilenet.MobileNetV3Large,
    }
    mobilenet_channels = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileNetV3Small': [16, 88, 144, 576],
        'MobileNetV3Large': [72, 120, 672, 960],
    }
    network = mobilenet_models[str(model_name)](width_multiplier=1.0)

    inputs = tf.ones([1, input_size, input_size, 3])
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobilenet_channels[model_name]):
      self.assertAllEqual(
          [1, input_size / 2 ** (idx+2), input_size / 2 ** (idx+2), num_filter],
          endpoints['res'+str(idx+2)].shape.as_list())

  @parameterized.parameters('MobileNetV3Small', 'MobileNetV3Large')
  def test_mobilenetv3_rf2_construct_graph(self, model_name):
    tf.keras.backend.set_image_data_format('channels_last')
    input_size = 128

    mobilenet_models = {
        'MobileNetV3Small': mobilenet.MobileNetV3Small,
        'MobileNetV3Large': mobilenet.MobileNetV3Large,
    }
    mobilenet_channels = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileNetV3Small': [16, 88, 144, 288],
        'MobileNetV3Large': [72, 120, 672, 480],
    }
    network = mobilenet_models[str(model_name)](width_multiplier=1.0,
                                                reduce_last_block_filters=True)

    inputs = tf.ones([1, input_size, input_size, 3])
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobilenet_channels[model_name]):
      self.assertAllEqual(
          [1, input_size / 2 ** (idx+2), input_size / 2 ** (idx+2), num_filter],
          endpoints['res'+str(idx+2)].shape.as_list())

  @parameterized.product(
      model_name=['MobileNetV3Small', 'MobileNetV3Large'],
      output_stride=[4, 8, 16, 32])
  def test_mobilenetv3_atrous_endpoint_shape(self, model_name, output_stride):
    tf.keras.backend.set_image_data_format('channels_last')
    input_size = 321
    batch_size = 2

    mobilenet_models = {
        'MobileNetV3Small': mobilenet.MobileNetV3Small,
        'MobileNetV3Large': mobilenet.MobileNetV3Large,
    }
    stride_spatial_shapes_map = {
        4: [81, 81, 81, 81],
        8: [81, 41, 41, 41],
        16: [81, 41, 21, 21],
        32: [81, 41, 21, 11],
    }
    mobilenet_channels = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileNetV3Small': [16, 88, 144, 576],
        'MobileNetV3Large': [72, 120, 672, 960],
    }
    network = mobilenet_models[str(model_name)](
        width_multiplier=1.0,
        output_stride=output_stride)
    spatial_shapes = stride_spatial_shapes_map[output_stride]

    inputs = tf.ones([batch_size, input_size, input_size, 3])
    endpoints = network(inputs)

    for idx, num_filters in enumerate(mobilenet_channels[model_name]):
      expected_shape = [
          batch_size, spatial_shapes[idx], spatial_shapes[idx], num_filters
      ]
      self.assertAllEqual(
          expected_shape,
          endpoints['res'+str(idx+2)].shape.as_list())

  @parameterized.parameters('MobileNetV3Small', 'MobileNetV3Large')
  def test_mobilenet_reload_weights(self, model_name):
    tf.keras.backend.set_image_data_format('channels_last')
    mobilenet_models = {
        'MobileNetV3Small': mobilenet.MobileNetV3Small,
        'MobileNetV3Large': mobilenet.MobileNetV3Large,
    }

    tf.random.set_seed(0)
    pixel_inputs = test_utils.create_test_input(1, 320, 320, 3)

    network1 = mobilenet_models[model_name](
        width_multiplier=1.0,
        output_stride=32,
        name='m1')
    network1(pixel_inputs, False)
    outputs1 = network1(pixel_inputs, False)
    pixel_outputs = outputs1['res5']

    # Feature extraction at the normal network rate.
    network2 = mobilenet_models[model_name](
        width_multiplier=1.0,
        output_stride=32,
        name='m2')
    network2(pixel_inputs, False)
    # Make the two networks use the same weights.
    network2.set_weights(network1.get_weights())
    outputs2 = network2(pixel_inputs, False)
    expected = outputs2['res5']

    self.assertAllClose(network1.get_weights(), network2.get_weights(),
                        atol=1e-4, rtol=1e-4)
    self.assertAllClose(pixel_outputs, expected, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
