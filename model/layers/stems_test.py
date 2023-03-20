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

"""Tests for resnet_utils."""
import tensorflow as tf

from deeplab2.model.layers import stems
from deeplab2.utils import test_utils


class ResnetUtilsTest(tf.test.TestCase):

  def test_inception_stem_output_shape(self):
    batch = 2
    height, width = 65, 65
    input_tensor = test_utils.create_test_input(batch, height, width, 3)
    model = stems.InceptionSTEM()
    output_tensor = model(input_tensor)
    expected_height = (height - 1) / 2 + 1
    expected_width = (width - 1) / 2 + 1
    expected_channels = 128
    self.assertListEqual(
        output_tensor.get_shape().as_list(),
        [batch, expected_height, expected_width, expected_channels])


if __name__ == '__main__':
  tf.test.main()
