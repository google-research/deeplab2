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

"""Tests for coco_tools."""

from absl.testing import absltest
import numpy as np
from pycocotools import mask

from deeplab2.utils import coco_tools


class CocoToolsTest(absltest.TestCase):

  def testSingleImageDetectionMaskExport(self):
    masks = np.array(
        [[[1, 1,], [1, 1]],
         [[0, 0], [0, 1]],
         [[0, 0], [0, 0]]], dtype=np.uint8)
    classes = np.array([1, 2, 3], dtype=np.int32)
    scores = np.array([0.8, 0.2, 0.7], dtype=np.float32)
    coco_annotations = coco_tools.ExportSingleImageDetectionMasksToCoco(
        image_id='first_image',
        category_id_set=set([1, 2, 3]),
        detection_classes=classes,
        detection_scores=scores,
        detection_masks=masks)
    expected_counts = ['04', '31', '4']
    for i, mask_annotation in enumerate(coco_annotations):
      self.assertEqual(mask_annotation['segmentation']['counts'],
                       expected_counts[i])
      self.assertTrue(np.all(np.equal(mask.decode(
          mask_annotation['segmentation']), masks[i])))
      self.assertEqual(mask_annotation['image_id'], 'first_image')
      self.assertEqual(mask_annotation['category_id'], classes[i])
      self.assertAlmostEqual(mask_annotation['score'], scores[i])

  def testSingleImageGroundtruthExport(self):
    masks = np.array(
        [[[1, 1,], [1, 1]],
         [[0, 0], [0, 1]],
         [[0, 0], [0, 0]]], dtype=np.uint8)
    boxes = np.array([[0, 0, 1, 1],
                      [0, 0, .5, .5],
                      [.5, .5, 1, 1]], dtype=np.float32)
    coco_boxes = np.array([[0, 0, 1, 1],
                           [0, 0, .5, .5],
                           [.5, .5, .5, .5]], dtype=np.float32)
    classes = np.array([1, 2, 3], dtype=np.int32)
    is_crowd = np.array([0, 1, 0], dtype=np.int32)
    next_annotation_id = 1
    expected_counts = ['04', '31', '4']

    # Tests exporting without passing in is_crowd (for backward compatibility).
    coco_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
        image_id='first_image',
        category_id_set=set([1, 2, 3]),
        next_annotation_id=next_annotation_id,
        groundtruth_boxes=boxes,
        groundtruth_classes=classes,
        groundtruth_masks=masks)
    for i, annotation in enumerate(coco_annotations):
      self.assertEqual(annotation['segmentation']['counts'],
                       expected_counts[i])
      self.assertTrue(np.all(np.equal(mask.decode(
          annotation['segmentation']), masks[i])))
      self.assertTrue(np.all(np.isclose(annotation['bbox'], coco_boxes[i])))
      self.assertEqual(annotation['image_id'], 'first_image')
      self.assertEqual(annotation['category_id'], classes[i])
      self.assertEqual(annotation['id'], i + next_annotation_id)

    # Tests exporting with is_crowd.
    coco_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
        image_id='first_image',
        category_id_set=set([1, 2, 3]),
        next_annotation_id=next_annotation_id,
        groundtruth_boxes=boxes,
        groundtruth_classes=classes,
        groundtruth_masks=masks,
        groundtruth_is_crowd=is_crowd)
    for i, annotation in enumerate(coco_annotations):
      self.assertEqual(annotation['segmentation']['counts'],
                       expected_counts[i])
      self.assertTrue(np.all(np.equal(mask.decode(
          annotation['segmentation']), masks[i])))
      self.assertTrue(np.all(np.isclose(annotation['bbox'], coco_boxes[i])))
      self.assertEqual(annotation['image_id'], 'first_image')
      self.assertEqual(annotation['category_id'], classes[i])
      self.assertEqual(annotation['iscrowd'], is_crowd[i])
      self.assertEqual(annotation['id'], i + next_annotation_id)


if __name__ == '__main__':
  absltest.main()
