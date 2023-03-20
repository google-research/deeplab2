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

"""Tests for panoptic_quality metrics."""
import collections

from absl import logging
import numpy as np
import tensorflow as tf

from deeplab2.evaluation import panoptic_quality
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


class PanopticQualityMetricTest(tf.test.TestCase):

  def test_streaming_metric_on_single_image(self):
    max_instances_per_category = 1000
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

    pred_classes = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', _CLASS_COLOR_MAP)
    pred_instances = test_utils.read_test_image(
        'team_pred_instance.png', image_format='L')

    pq_obj = panoptic_quality.PanopticQuality(
        num_classes=3,
        max_instances_per_category=max_instances_per_category,
        ignored_label=0, offset=256*256)

    y_true = combine_maps(gt_classes, gt_instances, max_instances_per_category)
    y_pred = combine_maps(pred_classes, pred_instances,
                          max_instances_per_category)
    pq_obj.update_state(y_true, y_pred)
    result = pq_obj.result().numpy()
    self.assertAlmostEqual(result[0], 0.62156284, places=4)
    self.assertAlmostEqual(result[1], 0.64664984, places=4)
    self.assertAlmostEqual(result[2], 0.9666667, places=4)
    self.assertEqual(result[3], 4.)
    self.assertAlmostEqual(result[4], 0.5)
    self.assertEqual(result[5], 0.)

  def test_streaming_metric_on_multiple_images(self):
    num_classes = 7

    bird_gt_instance_class_map = {
        92: 5,
        176: 3,
        255: 4,
    }
    cat_gt_instance_class_map = {
        0: 0,
        255: 6,
    }
    team_gt_instance_class_map = {
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
    max_instances_per_category = 256
    test_image = collections.namedtuple(
        'TestImage',
        ['gt_class_map', 'gt_path', 'pred_inst_path', 'pred_class_path'])
    test_images = [
        test_image(bird_gt_instance_class_map, 'bird_gt.png',
                   'bird_pred_instance.png', 'bird_pred_class.png'),
        test_image(cat_gt_instance_class_map, 'cat_gt.png',
                   'cat_pred_instance.png', 'cat_pred_class.png'),
        test_image(team_gt_instance_class_map, 'team_gt_instance.png',
                   'team_pred_instance.png', 'team_pred_class.png'),
    ]

    gt_classes = []
    gt_instances = []
    pred_classes = []
    pred_instances = []
    for test_image in test_images:
      (image_gt_instances,
       image_gt_classes) = test_utils.panoptic_segmentation_with_class_map(
           test_image.gt_path, test_image.gt_class_map)
      gt_classes.append(image_gt_classes)
      gt_instances.append(image_gt_instances)

      pred_classes.append(
          test_utils.read_segmentation_with_rgb_color_map(
              test_image.pred_class_path, _CLASS_COLOR_MAP))
      pred_instances.append(
          test_utils.read_test_image(test_image.pred_inst_path,
                                     image_format='L'))

    pq_obj = panoptic_quality.PanopticQuality(
        num_classes=num_classes,
        max_instances_per_category=max_instances_per_category,
        ignored_label=0, offset=256*256)
    for pred_class, pred_instance, gt_class, gt_instance in zip(
        pred_classes, pred_instances, gt_classes, gt_instances):
      y_true = combine_maps(gt_class, gt_instance, max_instances_per_category)
      y_pred = combine_maps(pred_class, pred_instance,
                            max_instances_per_category)
      pq_obj.update_state(y_true, y_pred)
    result = pq_obj.result().numpy()

    self.assertAlmostEqual(result[0], 0.76855499, places=4)
    self.assertAlmostEqual(result[1], 0.7769174, places=4)
    self.assertAlmostEqual(result[2], 0.98888892, places=4)
    self.assertEqual(result[3], 2.)
    self.assertAlmostEqual(result[4], 1. / 6, places=4)
    self.assertEqual(result[5], 0.)

  def test_predicted_non_contiguous_ignore_label(self):
    max_instances_per_category = 256
    pq_obj = panoptic_quality.PanopticQuality(
        num_classes=3,
        max_instances_per_category=max_instances_per_category,
        ignored_label=9,
        offset=256 * 256)

    gt_class = [
        [0, 9, 9],
        [1, 2, 2],
        [1, 9, 9],
    ]
    gt_instance = [
        [0, 2, 2],
        [1, 0, 0],
        [1, 0, 0],
    ]
    y_true = combine_maps(
        np.array(gt_class), np.array(gt_instance), max_instances_per_category)
    logging.info('y_true=\n%s', y_true)

    pred_class = [
        [0, 0, 9],
        [1, 1, 1],
        [1, 9, 9],
    ]
    pred_instance = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ]
    y_pred = combine_maps(
        np.array(pred_class), np.array(pred_instance),
        max_instances_per_category)
    logging.info('y_pred=\n%s', y_pred)

    pq_obj.update_state(y_true, y_pred)
    result = pq_obj.result().numpy()

    # pq
    self.assertAlmostEqual(result[0], 2. / 9, places=4)
    # sq
    self.assertAlmostEqual(result[1], 1. / 3, places=4)
    # rq
    self.assertAlmostEqual(result[2], 2. / 9, places=4)
    # tp
    self.assertAlmostEqual(result[3], 1. / 3, places=4)
    # fn
    self.assertAlmostEqual(result[4], 2. / 3, places=4)
    # fp
    self.assertAlmostEqual(result[5], 2. / 3, places=4)


if __name__ == '__main__':
  tf.test.main()
