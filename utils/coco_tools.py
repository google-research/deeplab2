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

"""Wrappers and conversions for third party pycocotools.

This is derived from code in the Tensorflow Object Detection API:
https://github.com/tensorflow/models/tree/master/research/object_detection

Huang et. al. "Speed/accuracy trade-offs for modern convolutional object
detectors" CVPR 2017.
"""

from typing import Any, Collection, Dict, List, Optional, Union

import numpy as np
from pycocotools import mask


COCO_METRIC_NAMES_AND_INDEX = (
    ('Precision/mAP', 0),
    ('Precision/mAP@.50IOU', 1),
    ('Precision/mAP@.75IOU', 2),
    ('Precision/mAP (small)', 3),
    ('Precision/mAP (medium)', 4),
    ('Precision/mAP (large)', 5),
    ('Recall/AR@1', 6),
    ('Recall/AR@10', 7),
    ('Recall/AR@100', 8),
    ('Recall/AR@100 (small)', 9),
    ('Recall/AR@100 (medium)', 10),
    ('Recall/AR@100 (large)', 11)
)


def _ConvertBoxToCOCOFormat(box: np.ndarray) -> List[float]:
  """Converts a box in [ymin, xmin, ymax, xmax] format to COCO format.

  This is a utility function for converting from our internal
  [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API
  i.e., [xmin, ymin, width, height].

  Args:
    box: a [ymin, xmin, ymax, xmax] numpy array

  Returns:
    a list of floats representing [xmin, ymin, width, height]
  """
  return [float(box[1]), float(box[0]), float(box[3] - box[1]),
          float(box[2] - box[0])]


def ExportSingleImageGroundtruthToCoco(
    image_id: Union[int, str],
    next_annotation_id: int,
    category_id_set: Collection[int],
    groundtruth_boxes: np.ndarray,
    groundtruth_classes: np.ndarray,
    groundtruth_masks: np.ndarray,
    groundtruth_is_crowd: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
  """Exports groundtruth of a single image to COCO format.

  This function converts groundtruth detection annotations represented as numpy
  arrays to dictionaries that can be ingested by the COCO evaluation API. Note
  that the image_ids provided here must match the ones given to
  ExportSingleImageDetectionsToCoco. We assume that boxes and classes are in
  correspondence - that is: groundtruth_boxes[i, :], and
  groundtruth_classes[i] are associated with the same groundtruth annotation.

  In the exported result, "area" fields are always set to the foregorund area of
  the mask.

  Args:
    image_id: a unique image identifier either of type integer or string.
    next_annotation_id: integer specifying the first id to use for the
      groundtruth annotations. All annotations are assigned a continuous integer
      id starting from this value.
    category_id_set: A set of valid class ids. Groundtruth with classes not in
      category_id_set are dropped.
    groundtruth_boxes: numpy array (float32) with shape [num_gt_boxes, 4]
    groundtruth_classes: numpy array (int) with shape [num_gt_boxes]
    groundtruth_masks: uint8 numpy array of shape [num_detections, image_height,
      image_width] containing detection_masks.
    groundtruth_is_crowd: optional numpy array (int) with shape [num_gt_boxes]
      indicating whether groundtruth boxes are crowd.

  Returns:
    a list of groundtruth annotations for a single image in the COCO format.

  Raises:
    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers
  """

  if len(groundtruth_classes.shape) != 1:
    raise ValueError('groundtruth_classes is '
                     'expected to be of rank 1.')
  if len(groundtruth_boxes.shape) != 2:
    raise ValueError('groundtruth_boxes is expected to be of '
                     'rank 2.')
  if groundtruth_boxes.shape[1] != 4:
    raise ValueError('groundtruth_boxes should have '
                     'shape[1] == 4.')
  num_boxes = groundtruth_classes.shape[0]
  if num_boxes != groundtruth_boxes.shape[0]:
    raise ValueError('Corresponding entries in groundtruth_classes, '
                     'and groundtruth_boxes should have '
                     'compatible shapes (i.e., agree on the 0th dimension).'
                     'Classes shape: %d. Boxes shape: %d. Image ID: %s' % (
                         groundtruth_classes.shape[0],
                         groundtruth_boxes.shape[0], image_id))
  has_is_crowd = groundtruth_is_crowd is not None
  if has_is_crowd and len(groundtruth_is_crowd.shape) != 1:
    raise ValueError('groundtruth_is_crowd is expected to be of rank 1.')
  groundtruth_list = []
  for i in range(num_boxes):
    if groundtruth_classes[i] in category_id_set:
      iscrowd = groundtruth_is_crowd[i] if has_is_crowd else 0
      segment = mask.encode(np.asfortranarray(groundtruth_masks[i]))
      area = mask.area(segment)
      export_dict = {
          'id': next_annotation_id + i,
          'image_id': image_id,
          'category_id': int(groundtruth_classes[i]),
          'bbox': list(_ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])),
          'segmentation': segment,
          'area': area,
          'iscrowd': iscrowd
      }

      groundtruth_list.append(export_dict)
  return groundtruth_list


def ExportSingleImageDetectionMasksToCoco(
    image_id: Union[int, str], category_id_set: Collection[int],
    detection_masks: np.ndarray, detection_scores: np.ndarray,
    detection_classes: np.ndarray) -> List[Dict[str, Any]]:
  """Exports detection masks of a single image to COCO format.

  This function converts detections represented as numpy arrays to dictionaries
  that can be ingested by the COCO evaluation API. We assume that
  detection_masks, detection_scores, and detection_classes are in correspondence
  - that is: detection_masks[i, :], detection_classes[i] and detection_scores[i]
    are associated with the same annotation.

  Args:
    image_id: unique image identifier either of type integer or string.
    category_id_set: A set of valid class ids. Detections with classes not in
      category_id_set are dropped.
    detection_masks: uint8 numpy array of shape [num_detections, image_height,
      image_width] containing detection_masks.
    detection_scores: float numpy array of shape [num_detections] containing
      scores for detection masks.
    detection_classes: integer numpy array of shape [num_detections] containing
      the classes for detection masks.

  Returns:
    a list of detection mask annotations for a single image in the COCO format.

  Raises:
    ValueError: if (1) detection_masks, detection_scores and detection_classes
      do not have the right lengths or (2) if each of the elements inside these
      lists do not have the correct shapes or (3) if image_ids are not integers.
  """

  if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
    raise ValueError('All entries in detection_classes and detection_scores'
                     'expected to be of rank 1.')
  num_boxes = detection_classes.shape[0]
  if not num_boxes == len(detection_masks) == detection_scores.shape[0]:
    raise ValueError('Corresponding entries in detection_classes, '
                     'detection_scores and detection_masks should have '
                     'compatible lengths and shapes '
                     'Classes length: %d.  Masks length: %d. '
                     'Scores length: %d' % (
                         detection_classes.shape[0], len(detection_masks),
                         detection_scores.shape[0]
                     ))
  detections_list = []
  for i in range(num_boxes):
    if detection_classes[i] in category_id_set:
      detections_list.append({
          'image_id': image_id,
          'category_id': int(detection_classes[i]),
          'segmentation': mask.encode(np.asfortranarray(detection_masks[i])),
          'score': float(detection_scores[i])
      })
  return detections_list
