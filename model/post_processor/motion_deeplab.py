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

"""This file contains functions to post-process Motion-DeepLab results."""

from typing import Tuple

import tensorflow as tf


def assign_instances_to_previous_tracks(
    prev_centers: tf.Tensor,
    current_centers: tf.Tensor,
    heatmap: tf.Tensor,
    offsets: tf.Tensor,
    panoptic_map: tf.Tensor,
    next_id: tf.Tensor,
    label_divisor: int,
    sigma=7) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Greedy assignment of current centers to previous centers.

  Current centers are selected in decreasing order of confidence (heatmap
  scores). These centers are transformed with the offsets and assigned to
  previous centers.

  Args:
    prev_centers: A tf.Tensor containing previous centers of shape [Np, 5]. This
      tensor contains:
      [0]: The x-coordinate.
      [1]: The y-coordinate.
      [2]: The panoptic ID.
      [3]: The geometric mean of width and height of the instance mask.
      [4]: The number of frames that no new masks got assigned to this center.
    current_centers: A tf.Tensor containing centers of current frame of shape
      [Nc, 5]. This tensor contains:
      [0]: The x-coordinate.
      [1]: The y-coordinate.
      [2]: The panoptic ID.
      [3]: The geometric mean of width and height of the instance mask.
      [4]: The number of frames that no new masks got assigned to this center.
    heatmap: A tf.Tensor of shape [batch, height, width] containing the center
      heatmap.
    offsets: A tf.Tensor of shape [batch, height, width, 2] containing the
      center offsets.
    panoptic_map: A tf.Tensor of shape [batch, height, width] containing the
      panoptic segmentation.
    next_id: A tf.Tensor of shape [1] containing the next ID.
    label_divisor: An integer specifying the label divisor for panoptic IDs.
    sigma: An optional integer specifying the number of frames that unmatched
      centers should be kept (default: 7).

  Returns:
    A tuple of three tf.Tensor:
      1. The updated panoptic segmentation map that contains track IDs.
      2. The updated tensor containing all current centers (including unmatched
        previous ones).
      3. The updated next ID that can be used for new tracks.
  """
  # Switch x and y coordinates for indexing.
  center_indices = tf.concat(
      [tf.zeros([tf.shape(current_centers)[0], 1], dtype=tf.int32),
       current_centers[:, 1:2], current_centers[:, 0:1]],
      axis=1)
  confidence_scores = tf.gather_nd(heatmap, center_indices)

  scores = tf.argsort(confidence_scores, direction='DESCENDING')
  cond = lambda i, *_: i < tf.shape(center_indices)[0]

  def body(i, current_centers_loop, prev_centers_loop, new_panoptic_map_loop,
           next_id_loop):
    row_index = scores[i]
    i = tf.add(i, 1)
    center_id = current_centers_loop[row_index, 2]
    center_location = current_centers_loop[row_index, :2]
    center_offset_yx = offsets[0, center_location[1], center_location[0], :]
    center_offset_xy = center_offset_yx[::-1]
    center_location = center_offset_xy + tf.cast(center_location, tf.float32)
    center_sem_id = center_id // label_divisor
    center_mask = tf.equal(panoptic_map, center_id)
    prev_centers_class = prev_centers_loop[:, 2] // label_divisor
    prev_centers_with_same_class = tf.squeeze(
        tf.cast(
            tf.gather(
                prev_centers_loop,
                tf.where(tf.equal(prev_centers_class, center_sem_id)),
                axis=0), tf.float32),
        axis=1)

    # Check if there are still unassigned previous centers of the same class.
    if tf.shape(prev_centers_with_same_class)[0] > 0:
      # For efficieny reasons, we do not take the sqrt when we compute the
      # minimal distances. See render_panoptic_map_as_heatmap as well.
      distances = tf.reduce_sum(
          tf.square(prev_centers_with_same_class[:, :2] - center_location),
          axis=1)
      prev_center_index = tf.math.argmin(
          distances, axis=0, output_type=tf.int32)
      min_dist = distances[prev_center_index]

      # If previous center is within a certain range, continue track.
      if min_dist < prev_centers_with_same_class[prev_center_index, 3]:
        new_center_id = tf.cast(
            prev_centers_with_same_class[prev_center_index, 2], dtype=tf.int32)
        shape = new_panoptic_map_loop.get_shape()
        new_panoptic_map_loop = tf.where(center_mask, new_center_id,
                                         new_panoptic_map_loop)
        new_panoptic_map_loop.set_shape(shape)
        current_centers_loop = tf.tensor_scatter_nd_update(
            current_centers_loop, tf.expand_dims([row_index, 2], 0),
            [new_center_id])
        # Remove previous center.
        prev_centers_loop = tf.squeeze(
            tf.gather(
                prev_centers_loop,
                tf.where(tf.not_equal(prev_centers_loop[:, 2], new_center_id)),
                axis=0),
            axis=1)
        return (i, current_centers_loop, prev_centers_loop,
                new_panoptic_map_loop, next_id_loop)
      else:
        # Assign new track ID
        new_center_id = center_sem_id * label_divisor + next_id_loop
        shape = new_panoptic_map_loop.get_shape()
        new_panoptic_map_loop = tf.where(center_mask, new_center_id,
                                         new_panoptic_map_loop)
        new_panoptic_map_loop.set_shape(shape)
        current_centers_loop = tf.tensor_scatter_nd_update(
            current_centers_loop, tf.expand_dims([row_index, 2], 0),
            [new_center_id])
        next_id_loop += 1
        return (i, current_centers_loop, prev_centers_loop,
                new_panoptic_map_loop, next_id_loop)
    else:
      # Assign new track ID
      new_center_id = center_sem_id * label_divisor + next_id_loop
      shape = new_panoptic_map_loop.get_shape()
      new_panoptic_map_loop = tf.where(center_mask, new_center_id,
                                       new_panoptic_map_loop)
      new_panoptic_map_loop.set_shape(shape)
      current_centers_loop = tf.tensor_scatter_nd_update(
          current_centers_loop, tf.expand_dims([row_index, 2], 0),
          [new_center_id])
      next_id_loop += 1
      return (i, current_centers_loop, prev_centers_loop, new_panoptic_map_loop,
              next_id_loop)

  loop_start_index = tf.constant(0)
  (_, current_centers,
   unmatched_centers, new_panoptic_map, next_id) = tf.while_loop(
       cond, body,
       (loop_start_index, current_centers, prev_centers, panoptic_map,
        next_id))

  # Keep unmatched centers for sigma frames.
  if tf.shape(unmatched_centers)[0] > 0:
    current_centers = tf.concat([current_centers, unmatched_centers], axis=0)

  number_centers = tf.shape(current_centers)[0]
  indices_row = tf.range(number_centers, dtype=tf.int32)
  indices_column = tf.repeat([4], number_centers, axis=0)
  indices = tf.stack([indices_row, indices_column], axis=1)
  current_centers = tf.tensor_scatter_nd_add(
      current_centers, indices,
      tf.repeat([1], number_centers, axis=0))

  # Remove centers after sigma frames.
  current_centers = tf.squeeze(
      tf.gather(
          current_centers,
          tf.where(tf.not_equal(current_centers[:, 4], sigma)),
          axis=0),
      axis=1)

  return new_panoptic_map, current_centers, next_id


def render_panoptic_map_as_heatmap(
    panoptic_map: tf.Tensor, sigma: int, label_divisor: int,
    void_label: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Extracts centers from panoptic map and renders as heatmap."""
  gaussian_size = 6 * sigma + 3
  x = tf.range(gaussian_size, dtype=tf.float32)
  y = tf.expand_dims(x, axis=1)
  x0, y0 = 3 * sigma + 1, 3 * sigma + 1
  gaussian = tf.math.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
  gaussian = tf.cast(tf.reshape(gaussian, [-1]), tf.float32)

  height = tf.shape(panoptic_map)[1]
  width = tf.shape(panoptic_map)[2]
  # Pad center to make boundary handling easier.
  center_pad_begin = int(round(3 * sigma + 1))
  center_pad_end = int(round(3 * sigma + 2))
  center_pad = center_pad_begin + center_pad_end

  center = tf.zeros((height + center_pad, width + center_pad))
  unique_ids, _ = tf.unique(tf.reshape(panoptic_map, [-1]))
  centers_and_ids = tf.TensorArray(
      tf.int32, size=0, dynamic_size=True, clear_after_read=False)
  counter = tf.zeros([], dtype=tf.int32)

  for panoptic_id in unique_ids:
    semantic_id = panoptic_id // label_divisor
    # Filter out IDs that should be ignored, are stuff classes or crowd.
    # Stuff classes and crowd regions both have IDs of the form panoptic_id =
    # semantic_id * label_divisor
    if semantic_id == void_label or panoptic_id % label_divisor == 0:
      continue

    # Convert [[0, y0, x0], ...] to [[0, ...], [y0, ...], [x0, ...]].
    mask_index = tf.cast(
        tf.transpose(tf.where(panoptic_map == panoptic_id)), tf.float32)
    mask_size = (
        tf.reduce_max(mask_index, axis=1) - tf.reduce_min(mask_index, axis=1))
    # The radius is defined as the geometric mean of width and height.
    # For efficieny reasons, we do not take the sqrt when we compute the minimal
    # distances. See assign_instances_to_previous_tracks as well.
    mask_radius = tf.cast(tf.round(mask_size[1] * mask_size[2]), tf.int32)
    centers = tf.reduce_mean(mask_index, axis=1)

    center_x = tf.cast(tf.round(centers[2]), tf.int32)
    center_y = tf.cast(tf.round(centers[1]), tf.int32)
    centers_and_ids = centers_and_ids.write(
        counter,
        [center_x, center_y, tf.cast(panoptic_id, tf.int32), mask_radius, 0])
    counter += 1

    # Due to the padding with center_pad_begin in center, the computed center
    # becomes the upper left corner in the center tensor.
    upper_left = center_x, center_y
    bottom_right = (upper_left[0] + gaussian_size,
                    upper_left[1] + gaussian_size)

    indices_x, indices_y = tf.meshgrid(
        tf.range(upper_left[0], bottom_right[0]),
        tf.range(upper_left[1], bottom_right[1]))
    indices = tf.transpose(
        tf.stack([tf.reshape(indices_y, [-1]),
                  tf.reshape(indices_x, [-1])]))

    center = tf.tensor_scatter_nd_max(
        center, indices, gaussian, name='center_scatter')

  center = center[center_pad_begin:(center_pad_begin + height),
                  center_pad_begin:(center_pad_begin + width)]
  return tf.expand_dims(center, axis=0), centers_and_ids.stack()
