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

"""Tests for activations.py."""
import tensorflow as tf

from deeplab2.model.layers import activations


class ActivationsTest(tf.test.TestCase):

  def test_gelu(self):
    expected_data = [[0.14967535, 0., -0.10032465],
                     [-0.15880796, -0.04540223, 2.9963627]]
    gelu_data = activations.gelu([[.25, 0, -.25], [-1, -2, 3]],
                                 approximate=True)
    self.assertAllClose(expected_data, gelu_data)
    gelu_data_via_get_activation = activations.get_activation(
        'approximated_gelu')([[.25, 0, -.25], [-1, -2, 3]])
    self.assertAllClose(expected_data, gelu_data_via_get_activation)


if __name__ == '__main__':
  tf.test.main()
