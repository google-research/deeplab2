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

"""Tests for (Axial-)ResNets."""


from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deeplab2.model.pixel_encoder import axial_resnet


class AxialResNetTest(tf.test.TestCase, parameterized.TestCase):

  # The parameter count does not include the classification head.
  @parameterized.parameters(
      ('resnet50', 23508032),
      ('axial_resnet50', 41343424),
  )
  def test_model_output_shape_and_num_params(self, model_name,
                                             expected_num_params):
    model = axial_resnet.get_model(model_name,
                                   input_shape=(224, 224, 3))
    output = model(tf.keras.Input(shape=(224, 224, 3)))

    if model_name == 'resnet50':
      dims = [64, 256, 512, 1024, 2048]
    elif model_name == 'axial_resnet50':
      dims = [128, 256, 512, 1024, 2048]

    self.assertListEqual(output['stage1'].get_shape().as_list(),
                         [None, 112, 112, dims[0]])
    self.assertListEqual(output['stage2'].get_shape().as_list(),
                         [None, 56, 56, dims[1]])
    self.assertListEqual(output['stage3'].get_shape().as_list(),
                         [None, 28, 28, dims[2]])
    self.assertListEqual(output['stage4'].get_shape().as_list(),
                         [None, 14, 14, dims[3]])
    self.assertListEqual(output['stage5'].get_shape().as_list(),
                         [None, 7, 7, dims[4]])

    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, expected_num_params)


if __name__ == '__main__':
  tf.test.main()
