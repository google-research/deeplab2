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

"""Tests for dataset_utils."""

import numpy as np
import tensorflow as tf

from deeplab2.data import dataset_utils


class DatasetUtilsTest(tf.test.TestCase):

  def _get_test_labels(self, num_classes, shape, label_divisor):
    num_ids_per_class = 35
    semantic_labels = np.random.randint(num_classes, size=shape)
    panoptic_labels = np.random.randint(
        num_ids_per_class, size=shape) + semantic_labels * label_divisor

    semantic_labels = tf.convert_to_tensor(semantic_labels, dtype=tf.int32)
    panoptic_labels = tf.convert_to_tensor(panoptic_labels, dtype=tf.int32)

    return panoptic_labels, semantic_labels

  def setUp(self):
    super().setUp()
    self._first_thing_class = 9
    self._num_classes = 19
    self._dataset_info = {
        'panoptic_label_divisor': 1000,
        'class_has_instances_list': tf.range(self._first_thing_class,
                                             self._num_classes)
    }
    self._num_ids = 37
    self._labels, self._semantic_classes = self._get_test_labels(
        self._num_classes, [2, 33, 33],
        self._dataset_info['panoptic_label_divisor'])

  def test_get_panoptic_and_semantic_label(self):
    # Note: self._labels contains one crowd instance per class.
    (returned_sem_labels, returned_pan_labels, returned_thing_mask,
     returned_crowd_region) = (
         dataset_utils.get_semantic_and_panoptic_label(
             self._dataset_info, self._labels, ignore_label=255))

    expected_semantic_labels = self._semantic_classes
    condition = self._labels % self._dataset_info['panoptic_label_divisor'] == 0
    condition = tf.logical_and(
        condition,
        tf.math.greater_equal(expected_semantic_labels,
                              self._first_thing_class))
    expected_crowd_labels = tf.where(condition, 1.0, 0.0)
    expected_pan_labels = tf.where(
        condition, 255 * self._dataset_info['panoptic_label_divisor'],
        self._labels)
    expected_thing_mask = tf.where(
        tf.math.greater_equal(expected_semantic_labels,
                              self._first_thing_class), 1.0, 0.0)

    self.assertListEqual(returned_sem_labels.shape.as_list(),
                         expected_semantic_labels.shape.as_list())
    self.assertListEqual(returned_pan_labels.shape.as_list(),
                         expected_pan_labels.shape.as_list())
    self.assertListEqual(returned_crowd_region.shape.as_list(),
                         expected_crowd_labels.shape.as_list())
    self.assertListEqual(returned_thing_mask.shape.as_list(),
                         expected_thing_mask.shape.as_list())
    np.testing.assert_equal(returned_sem_labels.numpy(),
                            expected_semantic_labels.numpy())
    np.testing.assert_equal(returned_pan_labels.numpy(),
                            expected_pan_labels.numpy())
    np.testing.assert_equal(returned_crowd_region.numpy(),
                            expected_crowd_labels.numpy())
    np.testing.assert_equal(returned_thing_mask.numpy(),
                            expected_thing_mask.numpy())

if __name__ == '__main__':
  tf.test.main()
