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

"""Tests for autoaugment_policy.py."""

import tensorflow as tf

from deeplab2.data.preprocessing import autoaugment_policy


class AutoaugmentPolicyTest(tf.test.TestCase):

  def testConvertPolicy(self):
    policy = [5, 1, 10, 5, 3, 4,
              6, 3, 7, 3, 3, 9,
              2, 2, 8, 8, 2, 8,
              1, 4, 9, 4, 5, 7,
              6, 4, 1, 1, 3, 4]
    expected = [
        [('Color', 0.2, 10), ('Color', 0.6, 4)],
        [('Contrast', 0.6, 7), ('Posterize', 0.6, 9)],
        [('Invert', 0.4, 8), ('Sharpness', 0.4, 8)],
        [('Equalize', 0.8, 9), ('Solarize', 1.0, 7)],
        [('Contrast', 0.8, 1), ('Equalize', 0.6, 4)],
    ]
    policy_list = autoaugment_policy.convert_policy(policy)
    self.assertAllEqual(policy_list, expected)


if __name__ == '__main__':
  tf.test.main()
