# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
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

import numpy as np


def stitch_video_panoptic_prediction(concat_panoptic: np.ndarray,
                                     next_panoptic: np.ndarray,
                                     label_divisor: int,
                                     overlap_offset: int = 128,
                                     combine_offset: int = 2**32) -> np.ndarray:
  """The stitching algorithm in ViP-DeepLab.

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

  def _ids_to_counts(id_array: np.ndarray):
    """Given a numpy array, a mapping from each entry to its count."""
    ids, counts = np.unique(id_array, return_counts=True)
    return dict(zip(ids, counts))

  new_panoptic = next_panoptic.copy()
  # Increase the panoptic instance ID to avoid overlap.
  new_category = new_panoptic // label_divisor
  new_instance = new_panoptic % label_divisor
  # We skip 0 which is reserved for crowd.
  instance_mask = new_instance > 0
  new_instance[instance_mask] = new_instance[instance_mask] + overlap_offset
  new_panoptic = new_category * label_divisor + new_instance
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
  # Match and propagate.
  for (concat_panoptic_label,
       next_panoptic_label) in map_concat_to_next.items():
    if map_next_to_concat[next_panoptic_label] == concat_panoptic_label:
      propagate_mask = next_panoptic == next_panoptic_label
      new_panoptic[propagate_mask] = concat_panoptic_label
  return new_panoptic
