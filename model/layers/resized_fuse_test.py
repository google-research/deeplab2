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

"""Tests for resized_fuse."""

import tensorflow as tf

from deeplab2.model.layers import resized_fuse


class ResizedFuseTest(tf.test.TestCase):

  def test_resize_and_fuse_features(self):
    batch, height, width, channels = 2, 11, 11, 6
    smaller_height, smaller_width, smaller_channels = 6, 6, 3
    larger_height1, larger_width1 = 21, 21  # Stride 2 conv.
    larger_height2, larger_width2 = 22, 22  # Stride 2 conv.
    larger_height3, larger_width3 = 23, 23  # Conv and resize.

    feature_list = []
    feature_list.append(tf.zeros([batch, smaller_height, smaller_width,
                                  smaller_channels]))
    feature_list.append(tf.zeros([batch, smaller_height, smaller_width,
                                  channels]))
    feature_list.append(tf.zeros([batch, height, width, smaller_channels]))
    feature_list.append(tf.zeros([batch, height, width, channels]))
    feature_list.append(tf.zeros([batch, larger_height1, larger_width1,
                                  channels]))
    feature_list.append(tf.zeros([batch, larger_height1, larger_width1,
                                  smaller_channels]))
    feature_list.append(tf.zeros([batch, larger_height2, larger_width2,
                                  smaller_channels]))
    feature_list.append(tf.zeros([batch, larger_height3, larger_width3,
                                  smaller_channels]))
    layer = resized_fuse.ResizedFuse(name='fuse',
                                     height=height,
                                     width=width,
                                     num_channels=channels)
    output = layer(feature_list)
    self.assertEqual(output.get_shape().as_list(), [batch, height, width,
                                                    channels])

if __name__ == '__main__':
  tf.test.main()
