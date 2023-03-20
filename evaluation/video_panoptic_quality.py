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

"""Implementation of the Video Panoptic Quality metric.

Video Panoptic Quality (VPQ) is an instance-based metric for evaluating the task
of video panoptic segmentation.
Please see the paper for details:
Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon.
"Video panoptic segmentation." In CVPR, 2020.

Note that our TensorFlow re-implementation of Video Panoptic Quality may give
a bit higher VPQ score than the official numpy implementation
(https://github.com/mcahny/vps).
This re-imeplementation is simply used as an estimate of the prediction
quality during TensorFlow training. The users should convert their video
panoptic prediction to COCO json format and run official numpy implementation,
in order to have a fair comparison with others. See Q4 in our FAQ (g3doc/faq.md)
for more details.
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from deeplab2.evaluation import panoptic_quality


class VideoPanopticQuality(panoptic_quality.PanopticQuality):
  """Metric class for Video Panoptic Quality.

  Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon.
  "Video panoptic segmentation." In CVPR, 2020.

  Video Panoptic Quality can be modeled as Image Panoptic Quality with the
  sequences of predictions and the ground-truth labels horizontally concatenated
  as two images, separately. Therefore, this class inherits the image panoptic
  quality class and changes the implementation to concatenated comparisons.

  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." In CVPR, 2021.

  Stand-alone usage:
  vpq_obj = video_panoptic_quality.VideoPanopticQuality(
    num_classes, max_instances_per_category, ignored_label)
  vpq_obj.update_state(y_true_1, y_pred_1)
  vpq_obj.update_state(y_true_2, y_pred_2)
  ...
  result = vpq_obj.result().numpy()
  """

  # pylint: disable=useless-super-delegation
  def __init__(self,
               num_classes: int,
               ignored_label: int,
               max_instances_per_category: int,
               offset: int,
               name: str = 'video_panoptic_quality',
               **kwargs):
    """Initialization of the VideoPanopticQuality metric.

    Args:
      num_classes: Number of classes in the dataset as an integer.
      ignored_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      max_instances_per_category: The maximum number of instances for each class
        as an integer or integer tensor.
      offset: The maximum number of unique labels as an integer or integer
        tensor.
      name: An optional variable_scope name. (default: 'video_panoptic_quality')
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super().__init__(num_classes, ignored_label, max_instances_per_category,
                     offset, name, **kwargs)

  def compare_and_accumulate(
      self, gt_panoptic_labels: List[tf.Tensor],
      pred_panoptic_labels: List[tf.Tensor]
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compares predicted segmentation with groundtruth, accumulates its metric.

    Args:
      gt_panoptic_labels: A list of tensors for the ground-truth video panoptic
        segmentation labels.
      pred_panoptic_labels: A list of tensors for video panoptic segmentation
        predictions.

    Returns:
      The value of the metrics (iou, tp, fn, fp) over all comparisons, as a
      float scalar.
    """
    gt_panoptic_label = tf.concat(gt_panoptic_labels, axis=1)
    pred_panoptic_label = tf.concat(pred_panoptic_labels, axis=1)
    return super(VideoPanopticQuality,
                 self).compare_and_accumulate(gt_panoptic_label,
                                              pred_panoptic_label)
