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

"""Tests for segmentation_and_tracking_quality.py.

This implementation is designed to work stand-alone. Please feel free to copy
this file and the corresponding segmentation_and_tracking_quality.py to your
project.
"""

import unittest
import numpy as np

import segmentation_and_tracking_quality as numpy_stq


def _compute_metric_and_compare(metric, ground_truth, prediction,
                                expected_result):
  metric.update_state(ground_truth, prediction, 1)
  result = metric.result()
  metric.reset_states()
  np.testing.assert_almost_equal(result['STQ'], expected_result[0])
  np.testing.assert_almost_equal(result['AQ'], expected_result[1])
  np.testing.assert_almost_equal(result['IoU'], expected_result[2])
  np.testing.assert_almost_equal(result['STQ_per_seq'], [expected_result[0]])
  np.testing.assert_almost_equal(result['AQ_per_seq'], [expected_result[1]])
  np.testing.assert_almost_equal(result['IoU_per_seq'], [expected_result[2]])
  np.testing.assert_almost_equal(result['ID_per_seq'], [1])
  np.testing.assert_almost_equal(result['Length_per_seq'], [1])


class STQualityTest(unittest.TestCase):

  def test_update_dict_stats(self):
    stat_dict = {}
    val = np.array([1, 1, 1, 2], dtype=np.int32)
    numpy_stq._update_dict_stats(stat_dict, val, None)
    self.assertDictEqual(stat_dict, {1: 3, 2: 1})

    stat_dict = {}
    val = np.array([1, 1, 1, 2], dtype=np.int32)
    weights = np.array([1, 0.5, 0.5, 0.5], dtype=np.float32)
    numpy_stq._update_dict_stats(stat_dict, val, weights)
    self.assertDictEqual(stat_dict, {1: 2, 2: 0.5})
    numpy_stq._update_dict_stats(stat_dict, val, weights)
    self.assertDictEqual(stat_dict, {1: 4, 2: 1})

  def test_complex_example(self):
    n_classes = 3
    ignore_label = 255
    # classes = ['sky', 'vegetation', 'cars'].
    things_list = [2]
    bit_shit = 16

    ground_truth_semantic_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 0, 1, 1, 1],
                                        [0, 2, 2, 2, 2, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 2, 1, 1],
                                        [2, 2, 2, 2, 2, 2, 2, 1],
                                        [2, 2, 2, 2, 2, 2, 2, 1],
                                        [2, 2, 2, 2, 2, 2, 1, 1]])
    ground_truth_semantic_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 2, 0, 0, 1, 1, 0, 0],
                                        [2, 2, 2, 1, 1, 1, 1, 0],
                                        [2, 2, 2, 2, 1, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 1, 1, 1],
                                        [2, 2, 2, 2, 1, 1, 1, 1]])
    ground_truth_semantic_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [2, 0, 1, 1, 1, 0, 0, 0],
                                        [2, 2, 1, 1, 1, 1, 0, 0],
                                        [2, 2, 2, 1, 1, 1, 1, 0],
                                        [2, 2, 2, 1, 1, 1, 1, 1],
                                        [2, 2, 2, 1, 1, 1, 1, 1]])
    ground_truth_semantic = np.stack([
        ground_truth_semantic_1, ground_truth_semantic_2,
        ground_truth_semantic_3
    ])

    ground_truth_instance_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 0, 0, 0, 0],
                                        [0, 2, 2, 2, 2, 0, 0, 0],
                                        [2, 2, 2, 2, 2, 2, 0, 0],
                                        [2, 2, 2, 2, 2, 2, 2, 0],
                                        [2, 2, 2, 2, 2, 2, 2, 0],
                                        [2, 2, 2, 2, 2, 2, 0, 0]])
    ground_truth_instance_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 2, 0, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 2, 0, 0, 0, 0],
                                        [2, 2, 2, 2, 2, 0, 0, 0],
                                        [2, 2, 2, 2, 2, 0, 0, 0],
                                        [2, 2, 2, 2, 0, 0, 0, 0]])
    ground_truth_instance_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [2, 0, 0, 0, 0, 0, 0, 0],
                                        [2, 2, 0, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 0, 0, 0, 0, 0]])

    ground_truth_instance = np.stack([
        ground_truth_instance_1, ground_truth_instance_2,
        ground_truth_instance_3
    ])
    ground_truth = ((ground_truth_semantic << bit_shit) + ground_truth_instance)

    prediction_semantic_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 2, 2, 1, 1, 1],
                                      [0, 2, 2, 2, 2, 2, 1, 1],
                                      [2, 2, 2, 2, 2, 2, 2, 1],
                                      [2, 2, 2, 2, 2, 2, 2, 1],
                                      [2, 2, 2, 2, 2, 2, 2, 1]])
    prediction_semantic_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 0, 0],
                                      [0, 2, 2, 2, 1, 1, 1, 1],
                                      [2, 2, 2, 2, 1, 1, 1, 1],
                                      [2, 2, 2, 2, 2, 1, 1, 1],
                                      [2, 2, 2, 2, 2, 2, 1, 1],
                                      [2, 2, 2, 2, 2, 1, 1, 1]])
    prediction_semantic_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 1, 0, 0],
                                      [2, 2, 2, 1, 1, 1, 0, 0],
                                      [2, 2, 2, 1, 1, 1, 1, 1],
                                      [2, 2, 2, 2, 1, 1, 1, 1],
                                      [2, 2, 2, 2, 1, 1, 1, 1]])
    prediction_semantic = np.stack(
        [prediction_semantic_1, prediction_semantic_2, prediction_semantic_3])

    prediction_instance_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 2, 2, 0, 0, 0],
                                      [0, 2, 2, 2, 2, 1, 0, 0],
                                      [2, 2, 2, 2, 2, 1, 1, 0],
                                      [2, 2, 2, 2, 1, 1, 1, 0],
                                      [2, 2, 2, 2, 1, 1, 1, 0]])
    prediction_instance_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 2, 2, 2, 0, 0, 0, 0],
                                      [2, 2, 2, 2, 0, 0, 0, 0],
                                      [2, 2, 2, 2, 2, 0, 0, 0],
                                      [2, 2, 2, 2, 1, 1, 0, 0],
                                      [2, 2, 2, 2, 1, 0, 0, 0]])
    prediction_instance_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 2, 2, 0, 0, 0, 0, 0],
                                      [2, 2, 2, 0, 0, 0, 0, 0],
                                      [2, 2, 2, 2, 0, 0, 0, 0],
                                      [2, 2, 2, 2, 0, 0, 0, 0]])
    prediction_instance = np.stack(
        [prediction_instance_1, prediction_instance_2, prediction_instance_3])
    prediction = ((prediction_semantic << bit_shit) + prediction_instance)

    # Compute STQuality.
    stq_metric = numpy_stq.STQuality(n_classes, things_list, ignore_label,
                                     bit_shit, 2**24)

    for i in range(3):
      stq_metric.update_state(ground_truth[i, ...].astype(dtype=np.int32),
                              prediction[i, ...].astype(dtype=np.int32), 1)

    result = stq_metric.result()

    np.testing.assert_almost_equal(result['STQ'], 0.66841773352)
    np.testing.assert_almost_equal(result['AQ'], 0.55366581415)
    np.testing.assert_almost_equal(result['IoU'], 0.8069529580309542)
    np.testing.assert_almost_equal(result['STQ_per_seq'], [0.66841773352])
    np.testing.assert_almost_equal(result['AQ_per_seq'], [0.55366581415])
    np.testing.assert_almost_equal(result['IoU_per_seq'], [0.8069529580309542])
    np.testing.assert_almost_equal(result['ID_per_seq'], [1])
    np.testing.assert_almost_equal(result['Length_per_seq'], [3])

  def test_basic_examples(self):
    n_classes = 2
    ignore_label = 255
    # classes = ['cars', 'sky'].
    things_list = [0]
    bit_shift = 16

    # Since the semantic label is `0`, the instance ID is enough.
    ground_truth_track = np.array([[1, 1, 1, 1, 1]])

    stq_metric = numpy_stq.STQuality(n_classes, things_list, ignore_label,
                                     bit_shift, 2**24)

    with self.subTest('Example 0'):
      predicted_track = np.array([[1, 1, 1, 1, 1]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [1.0, 1.0, 1.0])

    with self.subTest('Example 1'):
      predicted_track = np.array([[1, 1, 2, 2, 2]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.72111026, 0.52, 1.0])

    with self.subTest('Example 2'):
      predicted_track = np.array([[1, 2, 2, 2, 2]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.82462113, 0.68, 1.0])

    with self.subTest('Example 3'):
      predicted_track = np.array([[1, 2, 3, 4, 5]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.447213596, 0.2, 1.0])

    with self.subTest('Example 4'):
      predicted_track = np.array([[1, 2, 1, 2, 2]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.72111026, 0.52, 1.0])

    with self.subTest('Example 5'):
      predicted_track = (
          np.array([[0, 1, 1, 1, 1]]) +
          (np.array([[1, 0, 0, 0, 0]]) << bit_shift))
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.50596443, 0.64, 0.4])

    # First label is `crowd`.
    ground_truth_track = np.array([[0, 1, 1, 1, 1, 1]])

    with self.subTest('Example 6'):
      predicted_track = np.array([[1, 1, 1, 1, 1, 1]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [1.0, 1.0, 1.0])

    with self.subTest('Example 7'):
      predicted_track = np.array([[2, 2, 2, 2, 1, 1]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.72111026, 0.52, 1.0])

    with self.subTest('Example 8'):
      predicted_track = (
          np.array([[2, 2, 0, 1, 1, 1]]) +
          (np.array([[0, 0, 1, 0, 0, 0]]) << bit_shift))
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track,
                                  [0.40824829, 0.4, 5.0 / 12.0])

    # First label is `sky`.
    ground_truth_track = (
        np.array([[0, 1, 1, 1, 1]]) +
        (np.array([[1, 0, 0, 0, 0]]) << bit_shift))

    with self.subTest('Example 9'):
      predicted_track = np.array([[1, 1, 1, 1, 1]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.56568542, 0.8, 0.4])

    with self.subTest('Example 10'):
      predicted_track = np.array([[2, 2, 2, 1, 1]])
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.42426407, 0.45, 0.4])

    with self.subTest('Example 11'):
      predicted_track = (
          np.array([[2, 2, 0, 1, 1]]) +
          (np.array([[0, 0, 1, 0, 0]]) << bit_shift))
      _compute_metric_and_compare(stq_metric, ground_truth_track,
                                  predicted_track, [0.3, 0.3, 0.3])


if __name__ == '__main__':
  unittest.main()
