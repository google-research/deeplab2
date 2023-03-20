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

"""Tests of atrous consistencies for axial_resnet_instances."""

from absl.testing import parameterized
import tensorflow as tf

from deeplab2.model import test_utils
from deeplab2.model.encoder import axial_resnet_instances


class AtrousConsistencyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      (dict(model_name='resnet50', backbone_layer_multiplier=1),
       dict(model_name='resnet50_beta', backbone_layer_multiplier=1),
       dict(model_name='wide_resnet41', backbone_layer_multiplier=1),
       dict(model_name='swidernet', backbone_layer_multiplier=2)),
      output_stride=[8, 16, 32])
  def test_model_atrous_consistency_with_output_stride_four(
      self, model_name, backbone_layer_multiplier, output_stride):
    tf.random.set_seed(0)

    # Create the input.
    pixel_inputs = test_utils.create_test_input(1, 225, 225, 3)

    # Create the model and the weights.
    model_1 = axial_resnet_instances.get_model(
        model_name,
        # Test with small models only.
        num_blocks=[2, 2, 2, 2],
        backbone_layer_multiplier=backbone_layer_multiplier,
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001,
        output_stride=4)

    # Create the weights.
    model_1(pixel_inputs, training=False)

    # Set the batch norm gamma as non-zero so that the 3x3 convolution affects
    # the output.
    for weight in model_1.trainable_weights:
      if '/gamma:0' in weight.name:
        weight.assign(tf.ones_like(weight))

    # Dense feature extraction followed by subsampling.
    pixel_outputs = model_1(pixel_inputs, training=False)['res5']
    downsampling_stride = output_stride // 4
    expected = pixel_outputs[:, ::downsampling_stride, ::downsampling_stride, :]

    # Feature extraction at the nominal network rate.
    model_2 = axial_resnet_instances.get_model(
        model_name,
        # Test with small models only.
        num_blocks=[2, 2, 2, 2],
        backbone_layer_multiplier=backbone_layer_multiplier,
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001,
        output_stride=output_stride)
    # Create the weights.
    model_2(pixel_inputs, training=False)
    # Make the two networks use the same weights.
    model_2.set_weights(model_1.get_weights())
    output = model_2(pixel_inputs, training=False)['res5']

    # Normalize the outputs. Since we set batch_norm gamma to 1, the output
    # activations can explode to a large standard deviation, which sometimes
    # cause numerical errors beyond the tolerances.
    normalizing_factor = tf.math.reduce_std(expected)
    # Compare normalized outputs.
    self.assertAllClose(output / normalizing_factor,
                        expected / normalizing_factor,
                        atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
