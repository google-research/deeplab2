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

"""Tests for ConvNeXt."""


from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deeplab2.model.pixel_encoder import convnext


class ConvNeXtTest(tf.test.TestCase, parameterized.TestCase):

  # The parameter count does not include the classification head.
  @parameterized.parameters(
      ('convnext_tiny', 27818592),
      ('convnext_small', 49453152),
      ('convnext_base', 87564416),
      ('convnext_large', 196227264),
      ('convnext_xlarge', 348143872),
  )
  def test_model_output_shape_and_num_params(self, model_name,
                                             expected_num_params):
    model = convnext.get_model(model_name,
                               input_shape=(224, 224, 3))
    output = model(tf.keras.Input(shape=(224, 224, 3)))

    if model_name.lower() in ['convnext_tiny', 'convnext_small']:
      dims = [96, 192, 384, 768]
    elif model_name.lower() in ['convnext_base',]:
      dims = [128, 256, 512, 1024]
    elif model_name.lower() in ['convnext_large',]:
      dims = [192, 384, 768, 1536]
    elif model_name.lower() in ['convnext_xlarge',]:
      dims = [256, 512, 1024, 2048]

    self.assertListEqual(output['stage1'].get_shape().as_list(),
                         [None, 56, 56, dims[0]])
    self.assertListEqual(output['stage2'].get_shape().as_list(),
                         [None, 56, 56, dims[0]])
    self.assertListEqual(output['stage3'].get_shape().as_list(),
                         [None, 28, 28, dims[1]])
    self.assertListEqual(output['stage4'].get_shape().as_list(),
                         [None, 14, 14, dims[2]])
    self.assertListEqual(output['stage5'].get_shape().as_list(),
                         [None, 7, 7, dims[3]])

    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, expected_num_params)

  @parameterized.parameters(
      ('convnext_tiny', 224, 4383527995),
      ('convnext_small', 224, 8563618819),
      ('convnext_base', 224, 15194596739),
      ('convnext_large', 224, 34121222275),
      ('convnext_xlarge', 224, 60600740739),
      )
  def test_model_flops(self,
                       model_name,
                       input_resolution,
                       expected_multiply_adds):
    input_shape = [1, input_resolution, input_resolution, 3]
    model = convnext.get_model(model_name,
                               input_shape=input_shape[1:])
    model(tf.keras.Input(shape=input_shape[1:]))

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=input_shape)])

    graph_info = tf.compat.v1.profiler.profile(
        forward_pass.get_concrete_function().graph,
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    multiply_adds = graph_info.total_float_ops // 2
    self.assertEqual(multiply_adds, expected_multiply_adds)

if __name__ == '__main__':
  tf.test.main()
