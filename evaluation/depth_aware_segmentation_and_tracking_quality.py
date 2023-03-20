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

"""Implementation of Depth-aware Segmentation and Tracking Quality (DSTQ) metric."""

import collections
from typing import Sequence, List, Tuple
import tensorflow as tf
from deeplab2.evaluation import segmentation_and_tracking_quality as stq


class DSTQuality(stq.STQuality):
  """Metric class for Depth-aware Segmentation and Tracking Quality (DSTQ).

  This metric computes STQ and the inlier depth metric (or depth quality (DQ))
  under several thresholds. Then it returns the geometric mean of DQ's, AQ and
  IoU to get the final DSTQ, i.e.,

  DSTQ@{threshold_1} = pow(STQ ** 2 * DQ@{threshold_1}, 1/3)
  DSTQ@{threshold_2} = pow(STQ ** 2 * DQ@{threshold_2}, 1/3)
  ...
  DSTQ = pow(STQ ** 2 * DQ, 1/3)

  where DQ = pow(prod_i^n(threshold_i), 1/n) for n depth thresholds.

  The default choices for depth thresholds are 1.1 and 1.25, i.e.,
  max(pred/gt, gt/pred) <= 1.1 and max(pred/gt, gt/pred) <= 1.25.
  Commonly used thresholds for the inlier metrics are 1.25, 1.25**2, 1.25**3.
  These thresholds are so loose that many methods achieves > 99%.
  Therefore, we choose 1.25 and 1.1 to encourage high-precision predictions.

  Example usage:

  dstq_obj = depth_aware_segmentation_and_tracking_quality.DSTQuality(
    num_classes, things_list, ignore_label, max_instances_per_category,
    offset, depth_threshold)
  dstq.update_state(y_true_1, y_pred_1, d_true_1, d_pred_1)
  dstq.update_state(y_true_2, y_pred_2, d_true_2, d_pred_2)
  ...
  result = dstq_obj.result().numpy()
  """

  _depth_threshold: Tuple[float, float] = (1.25, 1.1)
  _depth_total_counts: collections.OrderedDict
  _depth_inlier_counts: List[collections.OrderedDict]

  def __init__(self,
               num_classes: int,
               things_list: Sequence[int],
               ignore_label: int,
               max_instances_per_category: int,
               offset: int,
               depth_threshold: Tuple[float] = (1.25, 1.1),
               name: str = 'dstq',):  # pytype: disable=annotation-type-mismatch
    """Initialization of the DSTQ metric.

    Args:
      num_classes: Number of classes in the dataset as an integer.
      things_list: A sequence of class ids that belong to `things`.
      ignore_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      max_instances_per_category: The maximum number of instances for each class
        as an integer or integer tensor.
      offset: The maximum number of unique labels as an integer or integer
        tensor.
      depth_threshold: A sequence of depth thresholds for the depth quality.
        (default: (1.25, 1.1))
      name: An optional name. (default: 'dstq')
    """
    super().__init__(num_classes, things_list, ignore_label,
                     max_instances_per_category, offset, name)
    if not (isinstance(depth_threshold, tuple) or
            isinstance(depth_threshold, list)):
      raise TypeError('The type of depth_threshold must be tuple or list.')
    if not depth_threshold:
      raise ValueError('depth_threshold must be non-empty.')
    self._depth_threshold = tuple(depth_threshold)
    self._depth_total_counts = collections.OrderedDict()
    self._depth_inlier_counts = []
    for _ in range(len(self._depth_threshold)):
      self._depth_inlier_counts.append(collections.OrderedDict())

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   d_true: tf.Tensor,
                   d_pred: tf.Tensor,
                   sequence_id: int = 0):
    """Accumulates the depth-aware segmentation and tracking quality statistics.

    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      y_pred: The predicted panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      d_true: The ground-truth depth map for this video frame.
      d_pred: The predicted depth map for this video frame.
      sequence_id: The optional ID of the sequence the frames belong to. When no
        sequence is given, all frames are considered to belong to the same
        sequence (default: 0).
    """
    super().update_state(y_true, y_pred, sequence_id)
    # Valid depth labels contain positive values.
    d_valid_mask = d_true > 0
    d_valid_total = tf.reduce_sum(tf.cast(d_valid_mask, tf.int32))
    # Valid depth prediction is expected to contain positive values.
    d_valid_mask = tf.logical_and(d_valid_mask, d_pred > 0)
    d_valid_true = tf.boolean_mask(d_true, d_valid_mask)
    d_valid_pred = tf.boolean_mask(d_pred, d_valid_mask)
    inlier_error = tf.maximum(d_valid_pred / d_valid_true,
                              d_valid_true / d_valid_pred)
    # For each threshold, count the number of inliers.
    for threshold_index, threshold in enumerate(self._depth_threshold):
      num_inliers = tf.reduce_sum(tf.cast(inlier_error <= threshold, tf.int32))
      inlier_counts = self._depth_inlier_counts[threshold_index]
      inlier_counts[sequence_id] = (inlier_counts.get(sequence_id, 0) +
                                    int(num_inliers.numpy()))
    # Update the total counts of the depth labels.
    self._depth_total_counts[sequence_id] = (
        self._depth_total_counts.get(sequence_id, 0) +
        int(d_valid_total.numpy()))

  def result(self):
    """Computes the depth-aware segmentation and tracking quality.

    Returns:
      A dictionary containing:
        - 'STQ': The total STQ score.
        - 'AQ': The total association quality (AQ) score.
        - 'IoU': The total mean IoU.
        - 'STQ_per_seq': A list of the STQ score per sequence.
        - 'AQ_per_seq': A list of the AQ score per sequence.
        - 'IoU_per_seq': A list of mean IoU per sequence.
        - 'Id_per_seq': A list of sequence Ids to map list index to sequence.
        - 'Length_per_seq': A list of the length of each sequence.
        - 'DSTQ': The total DSTQ score.
        - 'DSTQ@thres': The total DSTQ score for threshold thres
        - 'DSTQ_per_seq@thres': A list of DSTQ score per sequence for thres.
        - 'DQ': The total DQ score.
        - 'DQ@thres': The total DQ score for threshold thres.
        - 'DQ_per_seq@thres': A list of DQ score per sequence for thres.
    """
    # Gather the results for STQ.
    stq_results = super().result()
    # Collect results for depth quality per sequecne and threshold.
    dq_per_seq_at_threshold = {}
    dq_at_threshold = {}
    for threshold_index, threshold in enumerate(self._depth_threshold):
      dq_per_seq_at_threshold[threshold] = [0] * len(self._ground_truth)
      total_count = 0
      inlier_count = 0
      # Follow the order of computing STQ by enumerating _ground_truth.
      for index, sequence_id in enumerate(self._ground_truth):
        sequence_inlier = self._depth_inlier_counts[threshold_index][
            sequence_id]
        sequence_total = self._depth_total_counts[sequence_id]
        if sequence_total > 0:
          dq_per_seq_at_threshold[threshold][
              index] = sequence_inlier / sequence_total
        total_count += sequence_total
        inlier_count += sequence_inlier
      if total_count == 0:
        dq_at_threshold[threshold] = 0
      else:
        dq_at_threshold[threshold] = inlier_count / total_count
    # Compute DQ as the geometric mean of DQ's at different thresholds.
    dq = 1
    for _, threshold in enumerate(self._depth_threshold):
      dq *= dq_at_threshold[threshold]
    dq = dq ** (1 / len(self._depth_threshold))
    dq_results = {}
    dq_results['DQ'] = dq
    for _, threshold in enumerate(self._depth_threshold):
      dq_results['DQ@{}'.format(threshold)] = dq_at_threshold[threshold]
      dq_results['DQ_per_seq@{}'.format(
          threshold)] = dq_per_seq_at_threshold[threshold]
    # Combine STQ and DQ to get DSTQ.
    dstq_results = {}
    dstq_results['DSTQ'] = (stq_results['STQ'] ** 2 * dq) ** (1/3)
    for _, threshold in enumerate(self._depth_threshold):
      dstq_results['DSTQ@{}'.format(threshold)] = (
          stq_results['STQ'] ** 2 * dq_at_threshold[threshold]) ** (1/3)
      dstq_results['DSTQ_per_seq@{}'.format(threshold)] = [
          (stq_result**2 * dq_result)**(1 / 3) for stq_result, dq_result in zip(
              stq_results['STQ_per_seq'], dq_per_seq_at_threshold[threshold])
      ]
    # Merge all the results.
    dstq_results.update(stq_results)
    dstq_results.update(dq_results)
    return dstq_results

  def reset_states(self):
    """Resets all states that accumulated data."""
    super().reset_states()
    self._depth_total_counts = collections.OrderedDict()
    self._depth_inlier_counts = []
    for _ in range(len(self._depth_threshold)):
      self._depth_inlier_counts.append(collections.OrderedDict())
