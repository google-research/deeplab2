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

"""This file contains functions to post-process ViP-DeepLab results."""

from typing import MutableMapping, Tuple

import numpy as np
import tensorflow as tf


def get_stitch_video_panoptic_prediction_info(
    concat_panoptic: np.ndarray,
    next_panoptic: np.ndarray,
    label_divisor: int,
    combine_offset: int = 2**32) -> MutableMapping[int, int]:
  """Prepares the information for the stitching algorithm in ViP-DeepLab.

  Args:
    concat_panoptic: Panoptic prediction of the next frame by concatenating it
      with the current frame.
    next_panoptic: Panoptic prediction of the next frame.
    label_divisor: An integer specifying the label divisor of the dataset.
    combine_offset: An integer offset to combine concat and next panoptic.

  Returns:
    A map from the next frame instance IDs to the concatenated instances IDs.
  """

  def _ids_to_counts(id_array: np.ndarray):
    """Given a numpy array, a mapping from each entry to its count."""
    ids, counts = np.unique(id_array, return_counts=True)
    return dict(zip(ids, counts))

  # Pre-compute areas for all the segments.
  concat_segment_areas = _ids_to_counts(concat_panoptic)
  next_segment_areas = _ids_to_counts(next_panoptic)
  # Combine concat_panoptic and next_panoptic.
  intersection_id_array = (
      concat_panoptic.astype(np.int64) * combine_offset +
      next_panoptic.astype(np.int64))
  intersection_areas = _ids_to_counts(intersection_id_array)
  # Compute IoU and sort them.
  intersection_ious = []
  for intersection_id, intersection_area in intersection_areas.items():
    concat_panoptic_label = int(intersection_id // combine_offset)
    next_panoptic_label = int(intersection_id % combine_offset)
    concat_category_label = concat_panoptic_label // label_divisor
    next_category_label = next_panoptic_label // label_divisor
    if concat_category_label != next_category_label:
      continue
    concat_instance_label = concat_panoptic_label % label_divisor
    next_instance_label = next_panoptic_label % label_divisor
    # We skip 0 which is reserved for crowd.
    if concat_instance_label == 0 or next_instance_label == 0:
      continue
    union = (
        concat_segment_areas[concat_panoptic_label] +
        next_segment_areas[next_panoptic_label] - intersection_area)
    iou = intersection_area / union
    intersection_ious.append([concat_panoptic_label, next_panoptic_label, iou])
  intersection_ious = sorted(intersection_ious, key=lambda e: e[2])
  # Build mapping and inverse mapping. Two-way mapping guarantees 1-to-1
  # matching.
  map_concat_to_next = {}
  map_next_to_concat = {}
  for (concat_panoptic_label, next_panoptic_label, iou) in intersection_ious:
    map_concat_to_next[concat_panoptic_label] = next_panoptic_label
    map_next_to_concat[next_panoptic_label] = concat_panoptic_label

  remap_info = {
      next_label: concat_label
      for (concat_label, next_label) in map_concat_to_next.items()
      if map_next_to_concat[next_label] == concat_label
  }
  return remap_info


def stitch_video_panoptic_prediction(concat_panoptic: np.ndarray,
                                     next_panoptic: np.ndarray,
                                     label_divisor: int,
                                     overlap_offset: int = 128,
                                     combine_offset: int = 2**32) -> np.ndarray:
  """The numpy implementation of the stitching algorithm in ViP-DeepLab.

  This function stitches a pair of image panoptic predictions to form video
  panoptic predictions by propagating instance IDs from concat_panoptic to
  next_panoptic based on IoU matching.

  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." CVPR, 2021.

  Args:
    concat_panoptic: Panoptic prediction of the next frame by concatenating it
      with the current frame.
    next_panoptic: Panoptic prediction of the next frame.
    label_divisor: An integer specifying the label divisor of the dataset.
    overlap_offset: An integer offset to avoid overlap between the IDs in
      next_panoptic and the propagated IDs from concat_panoptic.
    combine_offset: An integer offset to combine concat and next panoptic.

  Returns:
    Panoptic prediction of the next frame with the instance IDs propragated
      from the concatenated panoptic prediction.
  """
  remap_info = get_stitch_video_panoptic_prediction_info(
      concat_panoptic, next_panoptic, label_divisor, combine_offset)

  new_panoptic = next_panoptic.copy()
  # Increase the panoptic instance ID to avoid overlap.
  new_category = new_panoptic // label_divisor
  new_instance = new_panoptic % label_divisor
  # We skip 0 which is reserved for crowd.
  instance_mask = new_instance > 0
  new_instance[instance_mask] = new_instance[instance_mask] + overlap_offset
  new_panoptic = new_category * label_divisor + new_instance
  # Match and propagate.
  for (next_panoptic_label, concat_panoptic_label) in remap_info.items():
    propagate_mask = next_panoptic == next_panoptic_label
    new_panoptic[propagate_mask] = concat_panoptic_label
  return new_panoptic


class VideoPanopticPredictionStitcher(tf.keras.Model):
  """The TF implementation of the stitching algorithm in ViP-DeepLab.

  It stitches a pair of image panoptic predictions to form video
  panoptic predictions by propagating instance IDs from concat_panoptic to
  next_panoptic based on IoU matching.

  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." CVPR, 2021.
  """

  def __init__(self,
               label_divisor: int,
               combine_offset: int = 2**32,
               name: str = 'video_panoptic_prediction_stitcher'):
    """Initializes a TF video panoptic prediction stitcher.

    It also sets the overlap_offset to label_divisor // 2 to avoid overlap
    between IDs in next_panoptic and the propagated IDs from concat_panoptic.
    label_divisor // 2 gives equal space for the two frames.

    Args:
      label_divisor: An integer specifying the label divisor of the dataset.
      combine_offset: An integer offset to combine concat and next panoptic.
      name: A string specifying the model name.
    """
    super().__init__(name=name)
    self._label_divisor = label_divisor
    self._overlap_offset = label_divisor // 2
    self._combine_offset = combine_offset

  def _ids_to_counts(
      self, id_array: tf.Tensor) -> tf.lookup.experimental.MutableHashTable:
    """Given a tf Tensor, returns a mapping from its elements to their counts.

    Args:
      id_array: A tf.Tensor from which the function counts its elements.

    Returns:
      A MutableHashTable that maps the elements to their counts.
    """
    ids, _, counts = tf.unique_with_counts(tf.reshape(id_array, [-1]))
    table = tf.lookup.experimental.MutableHashTable(
        key_dtype=ids.dtype, value_dtype=counts.dtype, default_value=-1)
    table.insert(ids, counts)
    return table

  def _increase_instance_ids_by_offset(self, panoptic: tf.Tensor) -> tf.Tensor:
    """Increases instance IDs by self._overlap_offset.

    Args:
      panoptic: A tf.Tensor for the panoptic prediction.

    Returns:
      A tf.Tensor for paonptic prediction with increased instance ids.
    """
    category = panoptic // self._label_divisor
    instance = panoptic % self._label_divisor
    # We skip 0 which is reserved for crowd.
    instance_mask = tf.greater(instance, 0)
    tf.assert_less(
        tf.reduce_max(instance + self._overlap_offset),
        self._label_divisor,
        message='Any new instance IDs cannot exceed label_divisor.')
    instance = tf.where(instance_mask, instance + self._overlap_offset,
                        instance)
    return category * self._label_divisor + instance

  def _compute_and_sort_iou_between_panoptic_ids(
      self, panoptic_1: tf.Tensor,
      panoptic_2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes and sorts intersecting panoptic IDs by IoU.

    Args:
      panoptic_1: A tf.Tensor for the panoptic prediction for frame 1.
      panoptic_2: A tf.Tensor for the panoptic prediction for frame 2.

    Returns:
      A tuple of tf.Tensor storing the mapping ids between the two input frames
        with their IoUs in the ascending order.
    """
    segment_areas_1 = self._ids_to_counts(panoptic_1)
    segment_areas_2 = self._ids_to_counts(panoptic_2)
    intersection_id_array = (
        tf.cast(panoptic_1, tf.int64) * self._combine_offset +
        tf.cast(panoptic_2, tf.int64))
    intersection_areas_table = self._ids_to_counts(intersection_id_array)
    intersection_ids, intersection_areas = intersection_areas_table.export()
    panoptic_ids_1 = tf.cast(intersection_ids // self._combine_offset, tf.int32)
    panoptic_ids_2 = tf.cast(intersection_ids % self._combine_offset, tf.int32)
    category_ids_1 = panoptic_ids_1 // self._label_divisor
    category_ids_2 = panoptic_ids_2 // self._label_divisor
    instance_ids_1 = panoptic_ids_1 % self._label_divisor
    instance_ids_2 = panoptic_ids_2 % self._label_divisor
    unions = (
        segment_areas_1.lookup(panoptic_ids_1) +
        segment_areas_2.lookup(panoptic_ids_2) - intersection_areas)
    intersection_ious = intersection_areas / unions
    is_valid_intersection = tf.logical_and(
        tf.equal(category_ids_1, category_ids_2),
        tf.logical_and(
            tf.not_equal(instance_ids_1, 0), tf.not_equal(instance_ids_2, 0)))
    intersection_ious = tf.gather(intersection_ious,
                                  tf.where(is_valid_intersection)[:, 0])
    panoptic_ids_1 = tf.gather(panoptic_ids_1,
                               tf.where(is_valid_intersection)[:, 0])
    panoptic_ids_2 = tf.gather(panoptic_ids_2,
                               tf.where(is_valid_intersection)[:, 0])
    ious_indices = tf.argsort(intersection_ious)
    panoptic_ids_1 = tf.gather(panoptic_ids_1, ious_indices)
    panoptic_ids_2 = tf.gather(panoptic_ids_2, ious_indices)
    segment_areas_1.remove(segment_areas_1.export()[0])
    segment_areas_2.remove(segment_areas_2.export()[0])
    intersection_areas_table.remove(intersection_areas_table.export()[0])
    return panoptic_ids_1, panoptic_ids_2

  def _match_and_propagate_instance_ids(
      self, concat_panoptic: tf.Tensor, next_panoptic: tf.Tensor,
      concat_panoptic_ids: tf.Tensor,
      next_panoptic_ids: tf.Tensor) -> tf.Tensor:
    """Propagates instance ids based on instance id matching.

    It propagates the instance ids from concat_panoptic to next_panoptic based
    on the mapping specified by concat_panoptic_ids and next_panoptic_ids.

    Args:
      concat_panoptic: A tf.Tensor for the concat panoptic prediction.
      next_panoptic: A tf.Tensor for the next panoptic prediction.
      concat_panoptic_ids: A tf.Tensor for the matching ids in concat_panoptic.
      next_panoptic_ids: A tf.Tensor for the matching ids in next_panoptic.

    Returns:
      A tf.Tensor for the next panoptic prediction with instance ids propagated
        from concat_panoptic.
    """
    map_concat_to_next = tf.lookup.experimental.MutableHashTable(
        key_dtype=tf.int32, value_dtype=tf.int32, default_value=-1)
    map_concat_to_next.insert(
        tf.cast(concat_panoptic_ids, tf.int32),
        tf.cast(next_panoptic_ids, tf.int32))
    map_next_to_concat = tf.lookup.experimental.MutableHashTable(
        key_dtype=tf.int32, value_dtype=tf.int32, default_value=-1)
    map_next_to_concat.insert(
        tf.cast(next_panoptic_ids, tf.int32),
        tf.cast(concat_panoptic_ids, tf.int32))
    matched_concat_panoptic_ids, matched_next_panoptic_ids = (
        map_concat_to_next.export())
    returned_concat_panoptic_ids = map_next_to_concat.lookup(
        matched_next_panoptic_ids)
    matched_ids_mask = tf.equal(matched_concat_panoptic_ids,
                                returned_concat_panoptic_ids)
    matched_concat_panoptic_ids = tf.gather(matched_concat_panoptic_ids,
                                            tf.where(matched_ids_mask)[:, 0])
    matched_next_panoptic_ids = tf.gather(matched_next_panoptic_ids,
                                          tf.where(matched_ids_mask)[:, 0])
    matched_concat_panoptic_ids = tf.expand_dims(
        tf.expand_dims(matched_concat_panoptic_ids, axis=-1), axis=-1)
    matched_next_panoptic_ids = tf.expand_dims(
        tf.expand_dims(matched_next_panoptic_ids, axis=-1), axis=-1)
    propagate_mask = tf.equal(next_panoptic, matched_next_panoptic_ids)
    panoptic_to_replace = tf.reduce_sum(
        tf.where(propagate_mask, matched_concat_panoptic_ids, 0),
        axis=0,
        keepdims=True)
    panoptic = tf.where(
        tf.reduce_any(propagate_mask, axis=0, keepdims=True),
        panoptic_to_replace, next_panoptic)
    panoptic = tf.ensure_shape(panoptic, next_panoptic.get_shape())
    map_concat_to_next.remove(map_concat_to_next.export()[0])
    map_next_to_concat.remove(map_next_to_concat.export()[0])
    return panoptic

  def call(self, concat_panoptic: tf.Tensor,
           next_panoptic: tf.Tensor) -> tf.Tensor:
    """Stitches the prediction from concat_panoptic and next_panoptic.

    Args:
      concat_panoptic: A tf.Tensor for the concat panoptic prediction.
      next_panoptic: A tf.Tensor for the next panoptic prediction.

    Returns:
      A tf.Tensor for the next panoptic prediction with instance ids propagated
        from concat_panoptic based on IoU matching.
    """
    next_panoptic = self._increase_instance_ids_by_offset(next_panoptic)
    concat_panoptic_ids, next_panoptic_ids = (
        self._compute_and_sort_iou_between_panoptic_ids(concat_panoptic,
                                                        next_panoptic))
    panoptic = self._match_and_propagate_instance_ids(concat_panoptic,
                                                      next_panoptic,
                                                      concat_panoptic_ids,
                                                      next_panoptic_ids)
    return panoptic
