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

"""Tests for input_preprocessing."""

import numpy as np
import tensorflow as tf

from deeplab2.data.preprocessing import input_preprocessing


class InputPreprocessingTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._image = tf.convert_to_tensor(np.random.randint(256, size=[33, 33, 3]))
    self._label = tf.convert_to_tensor(np.random.randint(19, size=[33, 33, 1]))

  def test_cropping(self):
    crop_height = np.random.randint(33)
    crop_width = np.random.randint(33)

    (original_image, processed_image, processed_label, prev_image, prev_label,
     _) = (
         input_preprocessing.preprocess_image_and_label(
             image=self._image,
             label=self._label,
             prev_image=tf.identity(self._image),
             prev_label=tf.identity(self._label),
             crop_height=crop_height,
             crop_width=crop_width,
             ignore_label=255))

    self.assertListEqual(original_image.shape.as_list(), [33, 33, 3])
    self.assertListEqual(processed_image.shape.as_list(),
                         [crop_height, crop_width, 3])
    self.assertListEqual(processed_label.shape.as_list(),
                         [crop_height, crop_width, 1])
    np.testing.assert_equal(processed_image.numpy(), prev_image.numpy())
    np.testing.assert_equal(processed_label.numpy(), prev_label.numpy())

  def test_resizing(self):
    height, width = 65, 65

    (original_image, processed_image, processed_label, prev_image, prev_label,
     _) = (
         input_preprocessing.preprocess_image_and_label(
             image=self._image,
             label=self._label,
             prev_image=tf.identity(self._image),
             prev_label=tf.identity(self._label),
             crop_height=height,
             crop_width=width,
             min_resize_value=65,
             max_resize_value=65,
             resize_factor=32,
             ignore_label=255))

    self.assertListEqual(original_image.shape.as_list(), [height, width, 3])
    self.assertListEqual(processed_image.shape.as_list(), [height, width, 3])
    self.assertListEqual(processed_label.shape.as_list(), [height, width, 1])
    np.testing.assert_equal(processed_image.numpy(), prev_image.numpy())
    np.testing.assert_equal(processed_label.numpy(), prev_label.numpy())

  def test_scaling(self):
    height, width = 65, 65

    (original_image, processed_image, processed_label, prev_image, prev_label,
     _) = (
         input_preprocessing.preprocess_image_and_label(
             image=self._image,
             label=self._label,
             prev_image=tf.identity(self._image),
             prev_label=tf.identity(self._label),
             crop_height=height,
             crop_width=width,
             min_scale_factor=0.5,
             max_scale_factor=2.0,
             ignore_label=255))

    self.assertListEqual(original_image.shape.as_list(), [33, 33, 3])
    self.assertListEqual(processed_image.shape.as_list(), [height, width, 3])
    self.assertListEqual(processed_label.shape.as_list(), [height, width, 1])
    np.testing.assert_equal(processed_image.numpy(), prev_image.numpy())
    np.testing.assert_equal(processed_label.numpy(), prev_label.numpy())

  def test_return_padded_image_and_label(self):
    image = np.dstack([[[5, 6], [9, 0]], [[4, 3], [3, 5]], [[7, 8], [1, 2]]])
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = np.array([[[1], [2]], [[3], [4]]])
    expected_image = np.dstack([[[127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 5, 6, 127.5, 127.5],
                                 [127.5, 9, 0, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5]],
                                [[127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 4, 3, 127.5, 127.5],
                                 [127.5, 3, 5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5]],
                                [[127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5],
                                 [127.5, 7, 8, 127.5, 127.5],
                                 [127.5, 1, 2, 127.5, 127.5],
                                 [127.5, 127.5, 127.5, 127.5, 127.5]]])
    expected_label = np.array([[[255], [255], [255], [255], [255]],
                               [[255], [255], [255], [255], [255]],
                               [[255], [1], [2], [255], [255]],
                               [[255], [3], [4], [255], [255]],
                               [[255], [255], [255], [255], [255]]])

    padded_image, padded_label = input_preprocessing._pad_image_and_label(
        image, label, 2, 1, 5, 5, 255)
    np.testing.assert_allclose(padded_image.numpy(), expected_image)
    np.testing.assert_allclose(padded_label.numpy(), expected_label)

  def test_return_original_image_when_target_size_is_equal_to_image_size(self):
    height, width, _ = tf.shape(self._image)
    padded_image, _ = input_preprocessing._pad_image_and_label(
        self._image, None, 0, 0, height, width)
    np.testing.assert_allclose(padded_image.numpy(), self._image)

  def test_die_on_target_size_greater_than_image_size(self):
    height, width, _ = tf.shape(self._image)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = input_preprocessing._pad_image_and_label(self._image, None, 0, 0,
                                                   height, width - 1)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = input_preprocessing._pad_image_and_label(self._image, None, 0, 0,
                                                   height - 1, width)

  def test_die_if_target_size_not_possible_with_given_offset(self):
    height, width, _ = tf.shape(self._image)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = input_preprocessing._pad_image_and_label(self._image, None, 3, 3,
                                                   height + 2, width + 2)

  def test_set_min_resize_value_only_during_training(self):
    crop_height = np.random.randint(33)
    crop_width = np.random.randint(33)

    _, processed_image, _, _, _, _ = (
        input_preprocessing.preprocess_image_and_label(
            image=self._image,
            label=self._label,
            crop_height=crop_height,
            crop_width=crop_width,
            min_resize_value=[10],
            max_resize_value=None,
            ignore_label=255))

    self.assertListEqual(processed_image.shape.as_list(),
                         [crop_height, crop_width, 3])


if __name__ == '__main__':
  tf.test.main()
