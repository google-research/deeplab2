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

"""COCO-style instance segmentation evaluation metrics.

Implements a Keras interface to COCO API.
COCO API: github.com/cocodataset/cocoapi/
"""
from typing import Any, Collection, Mapping, Optional

from absl import logging
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf

from deeplab2.utils import coco_tools
from deeplab2.utils import panoptic_instances


def _unwrap_segmentation(seg):
  return {
      'size': list(seg['size']),
      'counts': seg['counts'],
  }


_ANNOTATION_CONVERSION = {
    'bbox': list,
    'segmentation': _unwrap_segmentation,
}


def _unwrap_annotation(ann: Mapping[str, Any]) -> Mapping[str, Any]:
  """Unwraps the objects in an COCO-style annotation dictionary.

  Logic within the Keras metric class wraps the objects within the ground-truth
  and detection annotations in ListWrapper and DictWrapper classes. On the other
  hand, the COCO API does strict type checking as part of determining which
  branch to use in comparing detections and segmentations. We therefore have
  to coerce the types from the wrapper to the built-in types that COCO is
  expecting.

  Args:
    ann: A COCO-style annotation dictionary that may contain ListWrapper and
      DictWrapper objects.

  Returns:
    The same annotation information, but with wrappers reduced to built-in
    types.
  """
  unwrapped_ann = {}
  for k in ann:
    if k in _ANNOTATION_CONVERSION:
      unwrapped_ann[k] = _ANNOTATION_CONVERSION[k](ann[k])
    else:
      unwrapped_ann[k] = ann[k]
  return unwrapped_ann


class InstanceAveragePrecision(tf.keras.metrics.Metric):
  """COCO evaluation metric class."""

  def __init__(self, name: str = 'instance_ap', **kwargs):
    """Constructs COCO evaluation class."""
    super(InstanceAveragePrecision, self).__init__(name=name, **kwargs)
    self.reset_states()

  def reset_states(self) -> None:
    """Reset COCO API object."""
    self.detections = []
    self.dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    self.image_id = 1
    self.next_groundtruth_annotation_id = 1
    self.category_ids = set()
    self.metric_values = None

  def evaluate(self) -> np.ndarray:
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [12] representing the
        coco-style evaluation metrics.
    """
    if not self.detections:
      logging.warn('No detections to evaluate.')
      return np.zeros([12], dtype=np.float32)

    self.dataset['categories'] = [{
        'id': int(category_id)
    } for category_id in self.category_ids]

    # Creates "unwrapped" copies of COCO json-style objects.
    dataset = {
        'images': self.dataset['images'],
        'categories': self.dataset['categories']
    }
    dataset['annotations'] = [
        _unwrap_annotation(ann) for ann in self.dataset['annotations']
    ]
    detections = [_unwrap_annotation(ann) for ann in self.detections]

    logging.info('Creating COCO objects for AP eval...')
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(detections)

    logging.info('Running COCO evaluation...')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    return np.array(coco_metrics, dtype=np.float32)

  def result(self) -> np.ndarray:
    """Return the instance segmentation metric values, computing them if needed.

    Returns:
      A float vector of 12 elements. The meaning of each element is (in order):

       0. AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
       1. AP @[ IoU=0.50      | area=   all | maxDets=100 ]
       2. AP @[ IoU=0.75      | area=   all | maxDets=100 ]
       3. AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
       4. AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
       5. AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
       6. AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
       7. AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
       8. AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
       9. AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
      10. AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      11, AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]

      Where: AP = Average Precision
             AR = Average Recall
             IoU = Intersection over Union. IoU=0.50:0.95 is the average of the
               metric over thresholds of 0.5 to 0.95 with increments of 0.05.

      The area thresholds mean that, for those entries, ground truth annotation
      with area outside the range is ignored.
        small:  [0**2, 32**2],
        medium: [32**2, 96**2]
        large:  [96**2, 1e5**2]
    """
    if not self.metric_values:
      self.metric_values = self.evaluate()
    return self.metric_values

  def update_state(self, groundtruth_boxes: tf.Tensor,
                   groundtruth_classes: tf.Tensor, groundtruth_masks: tf.Tensor,
                   groundtruth_is_crowd: tf.Tensor, detection_masks: tf.Tensor,
                   detection_scores: tf.Tensor,
                   detection_classes: tf.Tensor) -> None:
    """Update detection results and groundtruth data.

    Append detection results to self.detections to the aggregate results from
    all of the validation set. The groundtruth_data is parsed and added into a
    dictionary with the same format as COCO dataset, which can be used for
    evaluation.

    Args:
      groundtruth_boxes: tensor (float32) with shape [num_gt_annos, 4]
      groundtruth_classes: tensor (int) with shape [num_gt_annos]
      groundtruth_masks: tensor (uint8) with shape [num_gt_annos, image_height,
        image_width]
      groundtruth_is_crowd: tensor (bool) with shape [num_gt_annos]
      detection_masks: tensor (uint8) with shape [num_detections, image_height,
        image_width]
      detection_scores: tensor (float32) with shape [num_detections]
      detection_classes: tensor (int) with shape [num_detections]
    """
    # Reset the caching of result values.
    self.metric_values = None

    # Update known category ids.
    self.category_ids.update(groundtruth_classes.numpy())
    self.category_ids.update(detection_classes.numpy())

    # Add ground-truth annotations.
    groundtruth_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
        self.image_id,
        self.next_groundtruth_annotation_id,
        self.category_ids,
        groundtruth_boxes.numpy(),
        groundtruth_classes.numpy(),
        groundtruth_masks=groundtruth_masks.numpy(),
        groundtruth_is_crowd=groundtruth_is_crowd.numpy())
    self.next_groundtruth_annotation_id += len(groundtruth_annotations)

    # Add to set of images for which there are gt & detections
    # Infers image size from groundtruth masks.
    _, height, width = groundtruth_masks.shape
    self.dataset['images'].append({
        'id': self.image_id,
        'height': height,
        'width': width,
    })
    self.dataset['annotations'].extend(groundtruth_annotations)

    # Add predictions/detections.
    detection_annotations = coco_tools.ExportSingleImageDetectionMasksToCoco(
        self.image_id, self.category_ids, detection_masks.numpy(),
        detection_scores.numpy(), detection_classes.numpy())
    self.detections.extend(detection_annotations)

    self.image_id += 1


def _instance_masks(panoptic_label_map: tf.Tensor,
                    instance_panoptic_labels: tf.Tensor) -> tf.Tensor:
  """Constructs an array of masks for each instance in a panoptic label map.

  Args:
    panoptic_label_map: An integer tensor of shape `[image_height, image_width]`
      specifying the panoptic label at each pixel.
    instance_panoptic_labels: An integer tensor of shape `[num_instances]` that
      gives the label for each unique instance for which to compute masks.

  Returns:
    A boolean tensor of shape `[num_instances, image_height, image_width]` where
    each slice in the first dimension gives the mask for a single instance over
    the entire image.
  """
  return tf.math.equal(
      tf.expand_dims(panoptic_label_map, 0),
      tf.reshape(instance_panoptic_labels,
                 [tf.size(instance_panoptic_labels), 1, 1]))


class PanopticInstanceAveragePrecision(tf.keras.metrics.Metric):
  """Computes instance segmentation AP of panoptic segmentations.

  Panoptic segmentation includes both "thing" and "stuff" classes. This class
  ignores the "stuff" classes to report metrics on only the "thing" classes
  that have discrete instances. It computes a series of AP-based metrics using
  the COCO evaluation scripts.
  """

  def __init__(self,
               num_classes: int,
               things_list: Collection[int],
               label_divisor: int,
               ignored_label: int,
               name: str = 'panoptic_instance_ap',
               **kwargs):
    """Constructs panoptic instance segmentation evaluation class."""
    super(PanopticInstanceAveragePrecision, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.stuff_list = set(range(num_classes)).difference(things_list)
    self.label_divisor = label_divisor
    self.ignored_label = ignored_label
    self.detection_metric = InstanceAveragePrecision()
    self.reset_states()

  def reset_states(self) -> None:
    self.detection_metric.reset_states()

  def result(self) -> np.ndarray:
    return self.detection_metric.result()

  def update_state(self,
                   groundtruth_panoptic: tf.Tensor,
                   predicted_panoptic: tf.Tensor,
                   semantic_probability: tf.Tensor,
                   instance_score_map: tf.Tensor,
                   is_crowd_map: Optional[tf.Tensor] = None) -> None:
    """Adds the results from a new image to be computed by the metric.

    Args:
      groundtruth_panoptic: A 2D integer tensor, with the true panoptic label at
        each pixel.
      predicted_panoptic: 2D integer tensor with predicted panoptic labels to be
        evaluated.
      semantic_probability: An float tensor of shape `[image_height,
        image_width, num_classes]`. Specifies at each pixel the estimated
        probability distribution that that pixel belongs to each semantic class.
      instance_score_map: A 2D float tensor, where the pixels for an instance
        will have the probability of that being an instance.
      is_crowd_map: A 2D boolean tensor. Where it is True, the instance in that
        region is a "crowd" instance. It is assumed that all pixels in an
        instance will have the same value in this map. If set to None (the
        default), it will be assumed that none of the ground truth instances are
        crowds.
    """
    classes_to_ignore = tf.convert_to_tensor([self.ignored_label] +
                                             list(self.stuff_list), tf.int32)
    (gt_unique_labels,
     gt_box_coords) = panoptic_instances.instance_boxes_from_masks(
         groundtruth_panoptic, classes_to_ignore, self.label_divisor)
    gt_classes = tf.math.floordiv(gt_unique_labels, self.label_divisor)

    gt_masks = _instance_masks(groundtruth_panoptic, gt_unique_labels)

    if is_crowd_map is None:
      gt_is_crowd = tf.zeros(tf.shape(gt_classes), tf.bool)
    else:
      gt_is_crowd = panoptic_instances.per_instance_is_crowd(
          is_crowd_map, groundtruth_panoptic, gt_unique_labels)

    (pred_unique_labels,
     pred_scores) = panoptic_instances.combined_instance_scores(
         predicted_panoptic, semantic_probability, instance_score_map,
         self.label_divisor, self.ignored_label)

    # Filter out stuff and ignored label.
    pred_classes = tf.math.floordiv(pred_unique_labels, self.label_divisor)
    pred_class_is_ignored = tf.math.reduce_any(
        tf.math.equal(
            tf.expand_dims(pred_classes, 1),
            tf.expand_dims(classes_to_ignore, 0)),
        axis=1)
    pred_class_is_kept = tf.math.logical_not(pred_class_is_ignored)
    pred_unique_labels = tf.boolean_mask(pred_unique_labels, pred_class_is_kept)
    pred_scores = tf.boolean_mask(pred_scores, pred_class_is_kept)

    # Recompute class labels after the filtering.
    pred_classes = tf.math.floordiv(pred_unique_labels, self.label_divisor)
    pred_masks = _instance_masks(predicted_panoptic, pred_unique_labels)

    self.detection_metric.update_state(gt_box_coords, gt_classes, gt_masks,
                                       gt_is_crowd, pred_masks, pred_scores,
                                       pred_classes)
