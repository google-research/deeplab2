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

"""Tests for axial_resnet."""

import numpy as np
import tensorflow as tf

from deeplab2.model.encoder import axial_resnet


class AxialResNetTest(tf.test.TestCase):

  def test_axial_resnet_correct_output_shape(self):
    model = axial_resnet.AxialResNet('max_deeplab_s')
    endpoints = model(tf.zeros([2, 65, 65, 3]), training=False)
    self.assertListEqual(endpoints['backbone_output'].get_shape().as_list(),
                         [2, 5, 5, 2048])
    self.assertListEqual(
        endpoints['transformer_class_feature'].get_shape().as_list(),
        [2, 128, 256])
    self.assertListEqual(
        endpoints['transformer_mask_feature'].get_shape().as_list(),
        [2, 128, 256])
    self.assertListEqual(endpoints['feature_panoptic'].get_shape().as_list(),
                         [2, 17, 17, 256])
    self.assertListEqual(endpoints['feature_semantic'].get_shape().as_list(),
                         [2, 5, 5, 2048])
    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    self.assertEqual(num_params, 61726624)

if __name__ == '__main__':
  tf.test.main()
