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

"""Tests for test_utils."""

import tensorflow as tf

from deeplab2.model import test_utils


class TestUtilsTest(tf.test.TestCase):

  def test_create_test_input(self):
    input_shape = [1, 2, 3, 4]
    input_tensor = test_utils.create_test_input(*input_shape)
    self.assertListEqual(input_tensor.get_shape().as_list(), input_shape)


if __name__ == '__main__':
  tf.test.main()
