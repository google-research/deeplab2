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

"""Tests for Tensorflow instance utils."""

import numpy as np
import tensorflow as tf

from deeplab2.utils import panoptic_instances


class TensorflowInstanceBoxesTest(tf.test.TestCase):

  def testFilterPanopticLabelsByIgnoreCategories(self):
    panoptic_divisor = 9
    panoptic_labels = [
        4 * panoptic_divisor + 3,
        2 * panoptic_divisor + 7,
        4 * panoptic_divisor,
        1 * panoptic_divisor + 2,
        4,
        8,
        4 * panoptic_divisor + 1,
        1 * panoptic_divisor,
    ]
    panoptic_labels_tensor = tf.constant(panoptic_labels, dtype=tf.int32)

    ignore_categories = [0, 4]
    is_kept = panoptic_instances.instances_without_ignore_categories(
        panoptic_labels_tensor, ignore_categories, panoptic_divisor)
    with self.cached_session() as sess:
      is_kept_result = sess.run(is_kept)

    np.testing.assert_array_equal(
        is_kept_result, [False, True, False, True, False, False, False, True])

  def testFilterByIgnoreCategoriesWithEmptyIgnore(self):
    panoptic_labels = [14, 19, 0, 2]
    panoptic_labels_tensor = tf.constant(panoptic_labels, dtype=tf.int32)
    panoptic_divisor = 7

    is_kept_empty_ignore = panoptic_instances.instances_without_ignore_categories(
        panoptic_labels_tensor, [], panoptic_divisor)
    with self.cached_session() as sess:
      is_kept_empty_ignore_result = sess.run(is_kept_empty_ignore)

    np.testing.assert_array_equal(is_kept_empty_ignore_result,
                                  [True, True, True, True])

  def testFilterByIgnoreCategoriesWithEmptyPanopticLabels(self):
    panoptic_labels = tf.zeros([0], dtype=tf.int32)
    ignore_categories = [2, 3]
    panoptic_divisor = 7

    is_kept_empty_labels = panoptic_instances.instances_without_ignore_categories(
        panoptic_labels, ignore_categories, panoptic_divisor)
    with self.cached_session() as sess:
      is_kept_empty_labels_result = sess.run(is_kept_empty_labels)
    np.testing.assert_array_equal(is_kept_empty_labels_result, [])

  def testComputesInstanceBoxes(self):
    instance_labels = [
        [0, 1, 1, 0],
        [0, 1, 2, 2],
        [5, 1, 2, 2],
        [1, 1, 1, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]
    instance_labels_tensor = tf.constant(instance_labels, dtype=tf.int64)

    category_labels = [
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [0, 0, 0, 1],
        [3, 3, 3, 3],
    ]
    category_labels_tensor = tf.constant(category_labels, dtype=tf.int64)

    panoptic_divisor = 13
    panoptic_labels = (
        instance_labels_tensor + panoptic_divisor * category_labels_tensor)

    ignore_categories = tf.constant([0, 3], dtype=tf.int64)

    unique_labels, box_coords = panoptic_instances.instance_boxes_from_masks(
        panoptic_labels, ignore_categories, panoptic_divisor)
    with self.cached_session() as sess:
      unique_labels_result, box_coords_result = sess.run(
          [unique_labels, box_coords])

    np.testing.assert_array_equal(unique_labels_result, [14, 15, 18, 27])
    np.testing.assert_array_equal(
        box_coords_result,
        [
            [0, 1, 3, 3],  # Category 1, Instance 1
            [1, 2, 5, 4],  # Category 1, Instance 2
            [2, 0, 3, 1],  # Category 1, Instance 5
            [3, 0, 4, 4],  # Category 2, Instance 1
        ])

  def testIgnoresNothing(self):
    instance_labels = [
        [0, 1, 1, 0],
        [0, 1, 2, 2],
        [5, 1, 2, 2],
    ]
    instance_labels_tensor = tf.constant(instance_labels, dtype=tf.int64)

    category_labels = [
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    category_labels_tensor = tf.constant(category_labels, dtype=tf.int64)

    panoptic_divisor = 7
    panoptic_labels = (
        instance_labels_tensor + panoptic_divisor * category_labels_tensor)

    unique_labels, box_coords = panoptic_instances.instance_boxes_from_masks(
        panoptic_labels, panoptic_divisor=panoptic_divisor)
    with self.cached_session() as sess:
      unique_labels_result, box_coords_result = sess.run(
          [unique_labels, box_coords])

    np.testing.assert_array_equal(unique_labels_result, [0, 8, 9, 12])
    np.testing.assert_array_equal(
        box_coords_result,
        [
            [0, 0, 2, 4],  # Category 0, Instance 0
            [0, 1, 3, 3],  # Category 1, Instance 1
            [1, 2, 3, 4],  # Category 1, Instance 2
            [2, 0, 3, 1],  # Category 1, Instance 5
        ])

  def testIgnoresEverything(self):
    instance_labels = [
        [0, 1, 1, 0],
        [0, 1, 2, 2],
        [5, 1, 2, 2],
    ]
    instance_labels_tensor = tf.constant(instance_labels, dtype=tf.int64)

    category_labels = [
        [0, 1, 1, 0],
        [0, 1, 2, 2],
        [1, 1, 2, 2],
    ]
    category_labels_tensor = tf.constant(category_labels, dtype=tf.int64)

    panoptic_divisor = 11
    panoptic_labels = (
        instance_labels_tensor + panoptic_divisor * category_labels_tensor)

    ignore_categories = [0, 1, 2]

    unique_labels, box_coords = panoptic_instances.instance_boxes_from_masks(
        panoptic_labels, ignore_categories, panoptic_divisor)
    with self.cached_session() as sess:
      unique_labels_result, box_coords_result = sess.run(
          [unique_labels, box_coords])

    self.assertSequenceEqual(unique_labels_result.shape, (0,))
    self.assertSequenceEqual(box_coords_result.shape, (0, 4))

  def testSingleInstance(self):
    instance_labels = [
        [0, 0, 0],
        [0, 0, 0],
    ]
    instance_labels_tensor = tf.constant(instance_labels, dtype=tf.int64)

    category_labels = [
        [3, 3, 3],
        [3, 3, 3],
    ]
    category_labels_tensor = tf.constant(category_labels, dtype=tf.int64)

    panoptic_divisor = 9
    panoptic_labels = (
        instance_labels_tensor + panoptic_divisor * category_labels_tensor)

    unique_labels, box_coords = panoptic_instances.instance_boxes_from_masks(
        panoptic_labels, panoptic_divisor=panoptic_divisor)
    with self.cached_session() as sess:
      unique_labels_result, box_coords_result = sess.run(
          [unique_labels, box_coords])

    np.testing.assert_array_equal(unique_labels_result, [27])
    np.testing.assert_array_equal(box_coords_result, [[0, 0, 2, 3]])


class InstanceScoringTest(tf.test.TestCase):

  def testGetsSemanticProbabilities(self):
    ignore_label = 3
    semantic_labels = [
        [0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ]
    semantic_tensor = tf.constant(semantic_labels, dtype=tf.int32)

    instances = [
        [1, 2, 2, 1, 1],
        [1, 2, 3, 3, 1],
        [6, 2, 3, 3, 3],
        [2, 2, 2, 2, 2],
    ]
    instances_tensor = tf.constant(instances, dtype=tf.int32)

    panoptic_divisor = 13
    panoptic_labels = semantic_tensor * panoptic_divisor + instances_tensor

    category_0_probability = [
        [1.0, 0.0, 0.0, 0.8, 0.9],
        [0.8, 0.0, 0.0, 0.2, 1.0],
        [0.1, 0.0, 0.2, 0.1, 0.2],
        [0.1, 0.1, 0.2, 0.0, 0.0],
    ]
    category_1_probability = [
        [0.0, 1.0, 0.9, 0.0, 0.1],
        [0.0, 1.0, 0.9, 0.9, 0.0],
        [0.8, 0.7, 0.7, 0.8, 0.9],
        [0.0, 0.2, 0.2, 0.0, 0.4],
    ]
    category_2_probability = (
        np.ones((4, 5), dtype=np.float32) -
        (np.array(category_0_probability) + np.array(category_1_probability)))
    semantic_probability = np.stack([
        category_0_probability, category_1_probability, category_2_probability
    ],
                                    axis=2)
    semantic_probability_tensor = tf.constant(semantic_probability, tf.float32)

    instance_panoptic_labels, _, instance_area = tf.unique_with_counts(
        tf.reshape(panoptic_labels, [20]))
    probs = panoptic_instances.per_instance_semantic_probabilities(
        panoptic_labels, instance_panoptic_labels, instance_area,
        semantic_probability_tensor, panoptic_divisor, ignore_label)

    probs_result = probs.numpy()

    np.testing.assert_array_almost_equal(probs_result,
                                         [0.9, 0.9, 0.84, 0.8, 0.76])

  def testCombineInstanceScores(self):
    # This test does not have any pixels equal to ignore_label, so a dummy value
    # is used as it's not being tested.
    ignore_label = -1
    semantic_labels = [
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    semantic_tensor = tf.constant(semantic_labels, dtype=tf.int32)

    instances = [
        [1, 2, 2, 1],
        [1, 2, 3, 3],
        [6, 2, 3, 3],
    ]
    instances_tensor = tf.constant(instances, dtype=tf.int32)

    panoptic_divisor = 8
    panoptic_labels = semantic_tensor * panoptic_divisor + instances_tensor

    category_0_probability = [
        [1.0, 0.0, 0.0, 0.8],
        [0.8, 0.0, 0.0, 0.2],
        [0.1, 0.0, 0.2, 0.1],
    ]
    category_1_probability = (
        np.ones((3, 4), dtype=np.float32) - np.array(category_0_probability))
    semantic_probability = np.stack([
        category_0_probability,
        category_1_probability,
    ],
                                    axis=2)
    semantic_probability_tensor = tf.constant(semantic_probability, tf.float32)

    instance_scores = [
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.7, 0.7],
        [0.8, 0.5, 0.7, 0.7],
    ]
    instance_scores_tensor = tf.constant(instance_scores, tf.float32)

    labels, combined_scores = panoptic_instances.combined_instance_scores(
        panoptic_labels, semantic_probability_tensor, instance_scores_tensor,
        panoptic_divisor, ignore_label)

    labels_result = labels.numpy()
    combined_scores_result = combined_scores.numpy()

    np.testing.assert_array_equal(labels_result, [1, 10, 11, 14])
    np.testing.assert_array_almost_equal(combined_scores_result,
                                         [0, 0.5, 0.875 * 0.7, 0.9 * 0.8])

  def testIgnoresLabel(self):
    # This ignore label will be outside the valid range of an index into the
    # last axis of the semantic probability tensor.
    ignore_label = 255
    semantic_labels = [
        [0, 1],
        [0, 255],
    ]
    semantic_tensor = tf.constant(semantic_labels, dtype=tf.int32)

    instances = [
        [1, 2],
        [1, 3],
    ]
    instances_tensor = tf.constant(instances, dtype=tf.int32)

    panoptic_divisor = 256
    panoptic_labels = semantic_tensor * panoptic_divisor + instances_tensor

    category_0_probability = [
        [1.0, 0.0],
        [0.8, 0.0],
    ]
    category_1_probability = (
        np.ones((2, 2), dtype=np.float32) - np.array(category_0_probability))
    semantic_probability = np.stack([
        category_0_probability,
        category_1_probability,
    ],
                                    axis=2)
    semantic_probability_tensor = tf.constant(semantic_probability, tf.float32)

    instance_scores = [
        [1.0, 0.5],
        [1.0, 0.5],
    ]
    instance_scores_tensor = tf.constant(instance_scores, tf.float32)

    labels, combined_scores = panoptic_instances.combined_instance_scores(
        panoptic_labels, semantic_probability_tensor, instance_scores_tensor,
        panoptic_divisor, ignore_label)

    labels_result = labels.numpy()
    combined_scores_result = combined_scores.numpy()

    np.testing.assert_array_equal(labels_result, [1, 258])
    np.testing.assert_array_almost_equal(combined_scores_result, [0.9, 0.5])


class InstanceIsCrowdTest(tf.test.TestCase):

  def testGetsIsCrowdValues(self):
    is_crowd_map = tf.constant([
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], tf.uint8)
    is_crowd_map = tf.cast(is_crowd_map, tf.bool)
    id_map = tf.constant([
        [0, 1, 1],
        [0, 2, 3],
        [4, 3, 3],
    ], tf.int32)
    output_ids = tf.range(5)

    instance_is_crowd = panoptic_instances.per_instance_is_crowd(
        is_crowd_map, id_map, output_ids)

    is_crowd_result = instance_is_crowd.numpy()
    np.testing.assert_array_equal(is_crowd_result,
                                  [True, False, False, True, False])

  def testGetsSubsetOfIsCrowdValues(self):
    is_crowd_map = tf.constant([
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], tf.uint8)
    is_crowd_map = tf.cast(is_crowd_map, tf.bool)
    id_map = tf.constant([
        [0, 1, 1],
        [0, 2, 3],
        [4, 3, 3],
    ], tf.int32)
    output_ids = [1, 3]

    instance_is_crowd = panoptic_instances.per_instance_is_crowd(
        is_crowd_map, id_map, output_ids)

    is_crowd_result = instance_is_crowd.numpy()
    np.testing.assert_array_equal(is_crowd_result, [False, True])

  def testGetsIsCrowdValuesWithIdsInArbitraryOrder(self):
    is_crowd_map = tf.constant([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ], tf.uint8)
    is_crowd_map = tf.cast(is_crowd_map, tf.bool)
    id_map = tf.constant([
        [0, 1, 1],
        [0, 2, 3],
        [4, 3, 3],
    ], tf.int32)
    output_ids = [1, 3, 0, 2, 4]

    instance_is_crowd = panoptic_instances.per_instance_is_crowd(
        is_crowd_map, id_map, output_ids)

    is_crowd_result = instance_is_crowd.numpy()
    np.testing.assert_array_equal(is_crowd_result,
                                  [False, True, True, False, True])


if __name__ == '__main__':
  tf.test.main()
