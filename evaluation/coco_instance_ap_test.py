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

"""Tests for the COCO Instance AP metric."""

from absl import logging
import numpy as np
import tensorflow as tf

from deeplab2.evaluation import coco_instance_ap
from deeplab2.evaluation import test_utils

# See the definition of the color names at:
#   https://en.wikipedia.org/wiki/Web_colors.
_CLASS_COLOR_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,  # Person (blue).
    (255, 0, 0): 2,  # Bear (red).
    (0, 255, 0): 3,  # Tree (lime).
    (255, 0, 255): 4,  # Bird (fuchsia).
    (0, 255, 255): 5,  # Sky (aqua).
    (255, 255, 0): 6,  # Cat (yellow).
}


def combine_maps(semantic_map, instance_map, label_divisor):
  combined_map = instance_map + semantic_map * label_divisor
  return tf.cast(combined_map, tf.int32)


class CocoInstanceApTest(tf.test.TestCase):

  def test_empty_evaluation(self):
    metric_obj = coco_instance_ap.InstanceAveragePrecision()
    result = metric_obj.result().numpy()
    # Empty evaluation returns all zeros as a placeholder.
    expected_result = np.zeros([12], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected_result)

  def test_evaluates_single_image(self):
    groundtruth_boxes = [
        [0.25, 0.4, 0.75, 1.0],
    ]
    groundtruth_classes = [8]
    groundtruth_masks = [[
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]]
    groundtruth_is_crowd = [False]

    detection_masks = [[
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]]
    detection_scores = [0.8]
    detection_classes = [8]

    groundtruth_boxes = tf.constant(groundtruth_boxes, dtype=tf.float32)
    groundtruth_classes = tf.constant(groundtruth_classes, dtype=tf.int32)
    groundtruth_masks = tf.constant(groundtruth_masks, dtype=tf.uint8)
    groundtruth_is_crowd = tf.constant(groundtruth_is_crowd, dtype=tf.bool)

    detection_masks = tf.constant(detection_masks, dtype=tf.uint8)
    detection_scores = tf.constant(detection_scores, dtype=tf.float32)
    detection_classes = tf.constant(detection_classes, dtype=tf.int32)

    metric_obj = coco_instance_ap.InstanceAveragePrecision()
    metric_obj.update_state(groundtruth_boxes, groundtruth_classes,
                            groundtruth_masks, groundtruth_is_crowd,
                            detection_masks, detection_scores,
                            detection_classes)
    result = metric_obj.result().numpy()

    # The IoU for the foreground match is 0.8. So it is a TP for 7/10 of the IoU
    # thresholds.
    expected_result = [0.7, 1, 1, 0.7, -1, -1, 0.7, 0.7, 0.7, 0.7, -1, -1]
    np.testing.assert_array_almost_equal(result, expected_result)


class PanopticInstanceApTest(tf.test.TestCase):

  def test_evaluates_single_image(self):
    num_classes = 3
    things_list = [1, 2]
    label_divisor = 256
    ignore_label = 0
    instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    gt_instances, gt_classes = test_utils.panoptic_segmentation_with_class_map(
        'team_gt_instance.png', instance_class_map)
    gt_panoptic = combine_maps(gt_classes, gt_instances, label_divisor)

    pred_classes = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', _CLASS_COLOR_MAP)
    pred_instances = test_utils.read_test_image(
        'team_pred_instance.png', image_format='L')

    pred_panoptic = combine_maps(pred_classes, pred_instances, label_divisor)
    semantic_probability = tf.ones(
        tf.concat([tf.shape(pred_panoptic), [num_classes]], 0))
    instance_score_map = tf.ones(tf.shape(pred_panoptic))

    metric_obj = coco_instance_ap.PanopticInstanceAveragePrecision(
        num_classes, things_list, label_divisor, ignore_label)
    metric_obj.update_state(gt_panoptic, pred_panoptic, semantic_probability,
                            instance_score_map)

    result = metric_obj.result().numpy()
    logging.info('result = %s', result)

    expected_result = [
        0.2549, 0.9356, 0.1215, -1.0, 0.2399, 0.501, 0.0812, 0.2688, 0.2688,
        -1.0, 0.2583, 0.5
    ]
    np.testing.assert_almost_equal(result, expected_result, decimal=4)

  def test_evaluates_with_scores(self):
    num_classes = 3
    things_list = list(range(num_classes))
    label_divisor = 256
    ignore_label = 0
    gt_classes = tf.constant([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    pred_classes = tf.constant([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    instances = tf.constant([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 3],
    ], tf.int32)

    gt_panoptic = combine_maps(gt_classes, instances, label_divisor)
    pred_panoptic = combine_maps(pred_classes, instances, label_divisor)

    semantic_probability = tf.constant([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
    ], tf.float32)
    semantic_probability = tf.transpose(semantic_probability, [1, 2, 0])

    # This score map gives higher score to the incorrect instance.
    bad_instance_scores = tf.constant([
        [0.4, 0.4, 0.9, 0.9],
        [0.4, 0.4, 0.9, 0.9],
        [0.0, 0.0, 0.8, 0.8],
        [0.0, 0.0, 0.8, 0.8],
    ], tf.float32)
    metric_obj = coco_instance_ap.PanopticInstanceAveragePrecision(
        num_classes, things_list, label_divisor, ignore_label)
    metric_obj.update_state(gt_panoptic, pred_panoptic, semantic_probability,
                            bad_instance_scores)

    bad_result = metric_obj.result().numpy()
    logging.info('bad_result = %s', bad_result)
    expected_bad_result = [
        0.5025, 0.5025, 0.5025, 0.5025, -1., -1., 0.25, 0.75, 0.75, 0.75, -1.,
        -1.
    ]
    np.testing.assert_almost_equal(bad_result, expected_bad_result, decimal=4)

    # This score map gives lower score to the incorrect instance.
    good_instance_scores = tf.constant([
        [0.9, 0.9, 0.4, 0.4],
        [0.9, 0.9, 0.4, 0.4],
        [0.0, 0.0, 0.8, 0.8],
        [0.0, 0.0, 0.8, 0.8],
    ], tf.float32)
    metric_obj.reset_states()
    metric_obj.update_state(gt_panoptic, pred_panoptic, semantic_probability,
                            good_instance_scores)

    good_result = metric_obj.result().numpy()
    logging.info('good_result = %s', good_result)

    # Since the correct instance(s) have higher score, the "good" scores should
    # give a result with higher AP.
    expected_good_result = [
        0.75248, 0.75248, 0.75248, 0.75248, -1, -1, 0.75, 0.75, 0.75, 0.75, -1,
        -1
    ]
    np.testing.assert_almost_equal(good_result, expected_good_result, decimal=4)

  def test_ignores_crowds(self):
    num_classes = 3
    things_list = list(range(num_classes))
    label_divisor = 256
    ignore_label = 0
    gt_classes = tf.constant([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    pred_classes = tf.constant([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    instances = tf.constant([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 3],
    ], tf.int32)
    is_crowd_map = tf.math.equal(instances, 2)

    gt_panoptic = combine_maps(gt_classes, instances, label_divisor)
    pred_panoptic = combine_maps(pred_classes, instances, label_divisor)

    semantic_probability = tf.ones(
        tf.concat([tf.shape(pred_panoptic), [num_classes]], 0))
    instance_score_map = tf.ones(tf.shape(pred_panoptic))

    metric_obj = coco_instance_ap.PanopticInstanceAveragePrecision(
        num_classes, things_list, label_divisor, ignore_label)
    metric_obj.update_state(gt_panoptic, pred_panoptic, semantic_probability,
                            instance_score_map, is_crowd_map)

    result = metric_obj.result().numpy()
    logging.info('result = %s', result)
    # Expect perfect results (for the quantities that have an AP value), because
    # the only mistake is a "crowd" instance.
    expected_result = [1., 1., 1., 1., -1., -1., 1., 1., 1., 1., -1., -1.]
    np.testing.assert_almost_equal(result, expected_result, decimal=4)

  def test_ignores_stuff(self):
    num_classes = 4
    things_list = [3]
    label_divisor = 256
    ignore_label = 0
    gt_classes = tf.constant([
        [3, 3, 2, 2],
        [3, 3, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    pred_classes = tf.constant([
        [3, 3, 1, 1],
        [3, 3, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], tf.int32)
    instances = tf.constant([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 3],
    ], tf.int32)

    gt_panoptic = combine_maps(gt_classes, instances, label_divisor)
    pred_panoptic = combine_maps(pred_classes, instances, label_divisor)

    semantic_probability = tf.ones(
        tf.concat([tf.shape(pred_panoptic), [num_classes]], 0))
    instance_score_map = tf.ones(tf.shape(pred_panoptic))

    metric_obj = coco_instance_ap.PanopticInstanceAveragePrecision(
        num_classes, things_list, label_divisor, ignore_label)
    metric_obj.update_state(gt_panoptic, pred_panoptic, semantic_probability,
                            instance_score_map)

    result = metric_obj.result().numpy()
    logging.info('result = %s', result)
    # Expect perfect results (for the quantities that have an AP value), because
    # the mistakes are all in "stuff" classes.
    expected_result = [1., 1., 1., 1., -1., -1., 1., 1., 1., 1., -1., -1.]
    np.testing.assert_almost_equal(result, expected_result, decimal=4)


if __name__ == '__main__':
  tf.test.main()
