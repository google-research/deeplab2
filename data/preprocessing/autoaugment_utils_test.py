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

"""Tests for autoaugment_utils.py."""

import numpy as np
import tensorflow as tf

from deeplab2.data.preprocessing import autoaugment_utils


class AutoaugmentUtilsTest(tf.test.TestCase):

  def testAugmentWithNamedPolicy(self):
    num_classes = 3
    np_image = np.random.randint(256, size=(13, 13, 3))
    image = tf.constant(np_image, dtype=tf.uint8)
    np_label = np.random.randint(num_classes, size=(13, 13, 1))
    label = tf.constant(np_label, dtype=tf.int32)
    image, label = autoaugment_utils.distort_image_with_autoaugment(
        image, label, ignore_label=255,
        augmentation_name='simple_classification_policy')
    self.assertTrue(image.numpy().any())
    self.assertTrue(label.numpy().any())


if __name__ == '__main__':
  tf.test.main()
