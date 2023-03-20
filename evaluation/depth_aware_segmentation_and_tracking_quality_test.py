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

"""Tests for depth_aware_segmentation_and_tracking_quality."""

import numpy as np
import tensorflow as tf

from deeplab2.evaluation import depth_aware_segmentation_and_tracking_quality as dstq


class DepthAwareSegmentationAndTrackingQualityTest(tf.test.TestCase):

  def test_complex_example(self):
    n_classes = 3
    ignore_label = 255
    # classes = ['sky', 'vegetation', 'cars'].
    things_list = [2]
    max_instances_per_category = 1000

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
    ground_truth = (ground_truth_semantic * max_instances_per_category
                    + ground_truth_instance)

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
    prediction = (prediction_semantic * max_instances_per_category
                  + prediction_instance)

    ground_truth_depth = np.array(
        [[56.1, 50.9, 54.0, 63.6, 68.6, 50.9, 50.9, 58.1],
         [62.6, 52.1, 00.0, 60.9, 62.4, 52.6, 56.3, 63.4],
         [57.1, 61.2, 63.8, 63.1, 52.3, 54.3, 52.1, 51.4],
         [65.8, 50.5, 58.9, 54.3, 00.0, 65.4, 63.8, 56.8],
         [50.6, 56.5, 53.0, 66.9, 51.8, 58.6, 65.9, 66.4],
         [53.5, 56.2, 53.6, 50.6, 64.6, 51.1, 68.7, 50.3],
         [69.0, 65.3, 66.4, 51.9, 68.3, 50.5, 00.0, 67.4],
         [59.7, 51.3, 50.1, 67.2, 68.8, 62.8, 64.9, 59.5]])
    prediction_depth = np.array(
        [[67.5, 36.9, 65.7, 77.9, 75.0, 45.1, 68.2, 63.3],
         [43.8, 63.0, 79.4, 78.1, 82.2, 36.9, 59.2, 83.2],
         [70.6, 73.2, 77.8, 71.3, 41.3, 47.5, 58.8, 64.8],
         [60.5, 51.7, 72.2, 49.8, 56.1, 60.7, 72.2, 73.0],
         [34.5, 55.7, 46.7, 47.4, 69.6, 43.5, 82.3, 84.8],
         [46.9, 39.5, 35.4, 61.3, 79.4, 42.2, 48.9, 56.3],
         [57.0, 75.0, 84.2, 46.3, 67.4, 55.5, 46.9, 70.0],
         [62.3, 58.3, 59.4, 74.5, 70.6, 54.6, 78.6, 48.1]])

    with self.subTest('No valid depth labels'):
      # Compute DSTQuality.
      dstq_metric = dstq.DSTQuality(
          n_classes, things_list, ignore_label, max_instances_per_category,
          256 * 256, (1.25, 1.1))
      no_valid_ground_truth_depth = ground_truth_depth * 0

      for i in range(3):
        dstq_metric.update_state(
            tf.convert_to_tensor(ground_truth[i, ...], dtype=tf.int32),
            tf.convert_to_tensor(prediction[i, ...], dtype=tf.int32),
            tf.convert_to_tensor(no_valid_ground_truth_depth, dtype=tf.float32),
            tf.convert_to_tensor(prediction_depth, dtype=tf.float32),
            1)
      result = dstq_metric.result()

      # Check if additional implementations alter the STQ results.
      # The example is copied from the complex example for testing STQ.
      # The results are expected to be unchanged.
      np.testing.assert_almost_equal(result['STQ'], 0.66841773352)
      np.testing.assert_almost_equal(result['AQ'], 0.55366581415)
      np.testing.assert_almost_equal(result['IoU'], 0.8069529580309542)
      np.testing.assert_almost_equal(result['STQ_per_seq'], [0.66841773352])
      np.testing.assert_almost_equal(result['AQ_per_seq'], [0.55366581415])
      np.testing.assert_almost_equal(result['IoU_per_seq'],
                                     [0.8069529580309542])
      np.testing.assert_almost_equal(result['ID_per_seq'], [1])
      np.testing.assert_almost_equal(result['Length_per_seq'], [3])
      # As there is no valid depth labels, any depth metrics should be 0.
      np.testing.assert_almost_equal(result['DSTQ'], 0.0)
      np.testing.assert_almost_equal(result['DSTQ@1.1'], 0.0)
      np.testing.assert_almost_equal(result['DSTQ@1.25'], 0.0)
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.1'], [0.0])
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.25'], [0.0])
      np.testing.assert_almost_equal(result['DQ'], 0.0)
      np.testing.assert_almost_equal(result['DQ@1.1'], 0.0)
      np.testing.assert_almost_equal(result['DQ@1.25'], 0.0)
      np.testing.assert_almost_equal(result['DQ_per_seq@1.1'], [0.0])
      np.testing.assert_almost_equal(result['DQ_per_seq@1.25'], [0.0])

    with self.subTest('Default depth thresholds'):
      # Compute DSTQuality.
      dstq_metric = dstq.DSTQuality(
          n_classes, things_list, ignore_label, max_instances_per_category,
          256 * 256, (1.25, 1.1))

      for i in range(3):
        dstq_metric.update_state(
            tf.convert_to_tensor(ground_truth[i, ...], dtype=tf.int32),
            tf.convert_to_tensor(prediction[i, ...], dtype=tf.int32),
            tf.convert_to_tensor(ground_truth_depth, dtype=tf.float32),
            tf.convert_to_tensor(prediction_depth, dtype=tf.float32),
            1)

      result = dstq_metric.result()
      # Prepare groundtruth metrics.
      valid_depth_labels_total = np.sum(ground_truth_depth > 0)
      valid_depth_labels = ground_truth_depth[ground_truth_depth > 0]
      valid_depth_pred = prediction_depth[ground_truth_depth > 0]
      valid_depth_error = np.maximum(valid_depth_pred / valid_depth_labels,
                                     valid_depth_labels / valid_depth_pred)
      dq_1_1 = np.sum(valid_depth_error <= 1.1) / valid_depth_labels_total
      dq_1_25 = np.sum(valid_depth_error <= 1.25) / valid_depth_labels_total

      # Check if additional implementations alter the STQ results.
      # The example is copied from the complex example for testing STQ.
      # The results are expected to be unchanged.
      np.testing.assert_almost_equal(result['STQ'], 0.66841773352)
      np.testing.assert_almost_equal(result['AQ'], 0.55366581415)
      np.testing.assert_almost_equal(result['IoU'], 0.8069529580309542)
      np.testing.assert_almost_equal(result['STQ_per_seq'], [0.66841773352])
      np.testing.assert_almost_equal(result['AQ_per_seq'], [0.55366581415])
      np.testing.assert_almost_equal(result['IoU_per_seq'],
                                     [0.8069529580309542])
      np.testing.assert_almost_equal(result['ID_per_seq'], [1])
      np.testing.assert_almost_equal(result['Length_per_seq'], [3])
      # Results are checked by groundtruth or equations.
      np.testing.assert_almost_equal(result['DSTQ'] ** 3,
                                     result['STQ'] ** 2 * result['DQ'])
      np.testing.assert_almost_equal(result['DSTQ@1.1'] ** 3,
                                     result['STQ'] ** 2 * result['DQ@1.1'])
      np.testing.assert_almost_equal(result['DSTQ@1.25'] ** 3,
                                     result['STQ'] ** 2 * result['DQ@1.25'])
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.1'],
                                     [result['DSTQ@1.1']])
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.25'],
                                     [result['DSTQ@1.25']])
      np.testing.assert_almost_equal(result['DQ'] ** 2,
                                     result['DQ@1.1'] * result['DQ@1.25'])
      np.testing.assert_almost_equal(result['DQ@1.1'], dq_1_1)
      np.testing.assert_almost_equal(result['DQ@1.25'], dq_1_25)
      np.testing.assert_almost_equal(result['DQ_per_seq@1.1'],
                                     [result['DQ@1.1']])
      np.testing.assert_almost_equal(result['DQ_per_seq@1.25'],
                                     [result['DQ@1.25']])
      # Results are checked by real numbers.
      np.testing.assert_almost_equal(result['DSTQ'], 0.5552059833215103)
      np.testing.assert_almost_equal(result['DSTQ@1.1'], 0.45663565048742255)
      np.testing.assert_almost_equal(result['DSTQ@1.25'],
                                     0.6750539157136957)
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.1'],
                                     [0.45663565048742255])
      np.testing.assert_almost_equal(result['DSTQ_per_seq@1.25'],
                                     [0.6750539157136957])
      np.testing.assert_almost_equal(result['DQ'], 0.3830597195261614)
      np.testing.assert_almost_equal(result['DQ@1.1'], 0.21311475409836064)
      np.testing.assert_almost_equal(result['DQ@1.25'], 0.6885245901639344)
      np.testing.assert_almost_equal(result['DQ_per_seq@1.1'],
                                     [0.21311475409836064])
      np.testing.assert_almost_equal(result['DQ_per_seq@1.25'],
                                     [0.6885245901639344])


if __name__ == '__main__':
  tf.test.main()
