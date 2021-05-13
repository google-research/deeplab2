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

"""This file contains functions to post-process model results."""

from typing import Tuple

import tensorflow as tf

from deeplab2.model import utils
from deeplab2.tensorflow_ops.python.ops import merge_semantic_and_instance_maps_op as merge_ops


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


def get_semantic_predictions(semantic_logits: tf.Tensor) -> tf.Tensor:
  """Computes the semantic classes from the predictions.

  Args:
    semantic_logits: A tf.tensor of shape [batch, height, width, classes].

  Returns:
    A tf.Tensor containing the semantic class prediction of shape
      [batch, height, width].
  """
  return tf.argmax(semantic_logits, axis=-1, output_type=tf.int32)


def _get_instance_centers_from_heatmap(
    center_heatmap: tf.Tensor, center_threshold: float, nms_kernel_size: int,
    keep_k_centers: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a list of instance centers.

  Args:
    center_heatmap: A tf.Tensor of shape [height, width, 1].
    center_threshold: A float setting the threshold for the center heatmap.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep (K).
      Non-positive values will keep all centers.

  Returns:
    A tuple of
    - tf.Tensor of shape [N, 2] containing N center coordinates (after
      non-maximum suppression) in (y, x) order.
    - tf.Tensor of shape [height, width] containing the center heatmap after
      non-maximum suppression.
  """
  # Threshold center map.
  center_heatmap = tf.where(
      tf.greater(center_heatmap, center_threshold), center_heatmap, 0.0)

  # Non-maximum suppression.
  padded_map = utils.add_zero_padding(center_heatmap, nms_kernel_size, rank=3)
  pooled_center_heatmap = tf.keras.backend.pool2d(
      tf.expand_dims(padded_map, 0),
      pool_size=(nms_kernel_size, nms_kernel_size),
      strides=(1, 1),
      padding='valid',
      pool_mode='max')
  center_heatmap = tf.where(
      tf.equal(pooled_center_heatmap, center_heatmap), center_heatmap, 0.0)
  center_heatmap = tf.squeeze(center_heatmap, axis=[0, 3])

  # `centers` is of shape (N, 2) with (y, x) order of the second dimension.
  centers = tf.where(tf.greater(center_heatmap, 0.0))

  if keep_k_centers > 0 and tf.shape(centers)[0] > keep_k_centers:
    topk_scores, _ = tf.math.top_k(
        tf.reshape(center_heatmap, [-1]), keep_k_centers, sorted=False)
    centers = tf.where(tf.greater(center_heatmap, topk_scores[-1]))

  return centers, center_heatmap


def _find_closest_center_per_pixel(centers: tf.Tensor,
                                   center_offsets: tf.Tensor) -> tf.Tensor:
  """Assigns all pixels to their closest center.

  Args:
    centers: A tf.Tensor of shape [N, 2] containing N centers with coordinate
      order (y, x).
    center_offsets: A tf.Tensor of shape [height, width, 2].

  Returns:
    A tf.Tensor of shape [height, width] containing the index of the closest
      center, per pixel.
  """
  height = tf.shape(center_offsets)[0]
  width = tf.shape(center_offsets)[1]

  x_coord, y_coord = tf.meshgrid(tf.range(width), tf.range(height))
  coord = tf.stack([y_coord, x_coord], axis=-1)

  center_per_pixel = tf.cast(coord, tf.float32) + center_offsets

  # centers: [N, 2] -> [N, 1, 2].
  # center_per_pixel: [H, W, 2] -> [1, H*W, 2].
  centers = tf.cast(tf.expand_dims(centers, 1), tf.float32)
  center_per_pixel = tf.reshape(center_per_pixel, [height*width, 2])
  center_per_pixel = tf.expand_dims(center_per_pixel, 0)

  # distances: [N, H*W].
  distances = tf.norm(centers - center_per_pixel, axis=-1)

  return tf.reshape(tf.argmin(distances, axis=0), [height, width])


def _get_instances_from_heatmap_and_offset(
    semantic_segmentation: tf.Tensor, center_heatmap: tf.Tensor,
    center_offsets: tf.Tensor, center_threshold: float,
    thing_class_ids: tf.Tensor, nms_kernel_size: int,
    keep_k_centers: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the instance assignment per pixel.

  Args:
    semantic_segmentation: A tf.Tensor containing the semantic labels of shape
      [height, width].
    center_heatmap: A tf.Tensor of shape [height, width, 1].
    center_offsets: A tf.Tensor of shape [height, width, 2].
    center_threshold: A float setting the threshold for the center heatmap.
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep.
      Negative values will keep all centers.

  Returns:
    A tuple of:
    - tf.Tensor containing the instance segmentation (filtered with the `thing`
      segmentation from the semantic segmentation output) with shape
      [height, width].
    - tf.Tensor containing the processed centermap with shape [height, width].
    - tf.Tensor containing instance scores (where higher "score" is a reasonable
      signal of a higher confidence detection.) Will be of shape [height, width]
      with the score for a pixel being the score of the instance it belongs to.
      The scores will be zero for pixels in background/"stuff" regions.
  """
  thing_segmentation = tf.zeros_like(semantic_segmentation)
  for thing_id in thing_class_ids:
    thing_segmentation = tf.where(tf.equal(semantic_segmentation, thing_id),
                                  1,
                                  thing_segmentation)

  centers, processed_center_heatmap = _get_instance_centers_from_heatmap(
      center_heatmap, center_threshold, nms_kernel_size, keep_k_centers)
  if tf.shape(centers)[0] == 0:
    return (tf.zeros_like(semantic_segmentation), processed_center_heatmap,
            tf.zeros_like(processed_center_heatmap))

  instance_center_index = _find_closest_center_per_pixel(
      centers, center_offsets)
  # Instance IDs should start with 1. So we use the index into the centers, but
  # shifted by 1.
  instance_segmentation = tf.cast(instance_center_index, tf.int32) + 1

  # The value of the heatmap at an instance's center is used as the score
  # for that instance.
  instance_scores = tf.gather_nd(processed_center_heatmap, centers)
  tf.debugging.assert_shapes([
      (centers, ('N', 2)),
      (instance_scores, ('N',)),
  ])
  # This will map the instance scores back to the image space: where each pixel
  # has a value equal to the score of its instance.
  flat_center_index = tf.reshape(instance_center_index, [-1])
  instance_score_map = tf.gather(instance_scores, flat_center_index)
  instance_score_map = tf.reshape(instance_score_map,
                                  tf.shape(instance_segmentation))
  instance_score_map *= tf.cast(thing_segmentation, tf.float32)

  return (thing_segmentation * instance_segmentation, processed_center_heatmap,
          instance_score_map)


@tf.function
def get_panoptic_predictions(
    semantic_logits: tf.Tensor, center_heatmap: tf.Tensor,
    center_offsets: tf.Tensor, center_threshold: float,
    thing_class_ids: tf.Tensor, label_divisor: int, stuff_area_limit: int,
    void_label: int, nms_kernel_size: int, keep_k_centers: int,
    merge_semantic_and_instance_with_tf_op: bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the semantic class and instance ID per pixel.

  Args:
    semantic_logits: A tf.Tensor of shape [batch, height, width, classes].
    center_heatmap: A tf.Tensor of shape [batch, height, width, 1].
    center_offsets: A tf.Tensor of shape [batch, height, width, 2].
    center_threshold: A float setting the threshold for the center heatmap.
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    label_divisor: An integer specifying the label divisor of the dataset.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least.
    void_label: An integer specifying the void label.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep.
      Negative values will keep all centers.
    merge_semantic_and_instance_with_tf_op: Boolean, specifying the merging
      operation uses TensorFlow (CUDA kernel) implementation (True) or
      tf.py_function implementation (False). Note the tf.py_function
      implementation is simply used as a backup solution when you could not
      successfully compile the provided TensorFlow implementation. To reproduce
      our results, please use the provided TensorFlow implementation `merge_ops`
      (i.e., set to True).

  Returns:
    A tuple of:
    - the panoptic prediction as tf.Tensor with shape [batch, height, width].
    - the semantic prediction as tf.Tensor with shape [batch, height, width].
    - the instance prediction as tf.Tensor with shape [batch, height, width].
    - the centermap prediction as tf.Tensor with shape [batch, height, width].
    - the instance score maps as tf.Tensor with shape [batch, height, width].
  """
  semantic_prediction = get_semantic_predictions(semantic_logits)
  batch_size = tf.shape(semantic_logits)[0]

  instance_map_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  center_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  instance_score_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)

  for i in tf.range(batch_size):
    (instance_map, center_map,
     instance_score_map) = _get_instances_from_heatmap_and_offset(
         semantic_prediction[i, ...], center_heatmap[i, ...],
         center_offsets[i, ...], center_threshold, thing_class_ids,
         nms_kernel_size, keep_k_centers)
    instance_map_lists = instance_map_lists.write(i, instance_map)
    center_map_lists = center_map_lists.write(i, center_map)
    instance_score_map_lists = instance_score_map_lists.write(
        i, instance_score_map)

  # This does not work with unknown shapes.
  instance_maps = instance_map_lists.stack()
  center_maps = center_map_lists.stack()
  instance_score_maps = instance_score_map_lists.stack()

  if merge_semantic_and_instance_with_tf_op:
    panoptic_prediction = merge_ops.merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
  else:
    panoptic_prediction = _merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
  return (panoptic_prediction, semantic_prediction, instance_maps, center_maps,
          instance_score_maps)


@tf.function
def _merge_semantic_and_instance_maps(
    semantic_prediction: tf.Tensor,
    instance_maps: tf.Tensor,
    thing_class_ids: tf.Tensor,
    label_divisor: int,
    stuff_area_limit: int,
    void_label: int) -> tf.Tensor:
  """Merges semantic and instance maps to obtain panoptic segmentation.

  This function merges the semantic segmentation and class-agnostic
  instance segmentation to form the panoptic segmentation. In particular,
  the class label of each instance mask is inferred from the majority
  votes from the corresponding pixels in the semantic segmentation. This
  operation is first poposed in the DeeperLab paper and adopted by the
  Panoptic-DeepLab.

  - DeeperLab: Single-Shot Image Parser, T-J Yang, et al. arXiv:1902.05093.
  - Panoptic-DeepLab, B. Cheng, et al. In CVPR, 2020.

  Note that this function only supports batch = 1 for simplicity. Additionally,
  this function has a slightly different implementation from the provided
  TensorFlow implementation `merge_ops` but with a similar performance. This
  function is mainly used as a backup solution when you could not successfully
  compile the provided TensorFlow implementation. To reproduce our results,
  please use the provided TensorFlow implementation (i.e., not use this
  function, but the `merge_ops.merge_semantic_and_instance_maps`).

  Args:
    semantic_prediction: A tf.Tensor of shape [batch, height, width].
    instance_maps: A tf.Tensor of shape [batch, height, width].
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    label_divisor: An integer specifying the label divisor of the dataset.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least.
    void_label: An integer specifying the void label.

  Returns:
    panoptic_prediction: A tf.Tensor with shape [batch, height, width].
  """
  prediction_shape = semantic_prediction.get_shape().as_list()
  # This implementation only supports batch size of 1. Since model construction
  # might lose batch size information (and leave it to None), override it here.
  prediction_shape[0] = 1
  semantic_prediction = tf.ensure_shape(semantic_prediction, prediction_shape)
  instance_maps = tf.ensure_shape(instance_maps, prediction_shape)

  # Default panoptic_prediction to have semantic label = void_label.
  panoptic_prediction = tf.ones_like(
      semantic_prediction) * void_label * label_divisor

  # Start to paste predicted `thing` regions to panoptic_prediction.
  # Infer `thing` segmentation regions from semantic prediction.
  semantic_thing_segmentation = tf.zeros_like(semantic_prediction,
                                              dtype=tf.bool)
  for thing_class in thing_class_ids:
    semantic_thing_segmentation = tf.math.logical_or(
        semantic_thing_segmentation,
        semantic_prediction == thing_class)
  # Keep track of how many instances for each semantic label.
  num_instance_per_semantic_label = tf.TensorArray(
      tf.int32, size=0, dynamic_size=True, clear_after_read=False)
  instance_ids, _ = tf.unique(tf.reshape(instance_maps, [-1]))
  for instance_id in instance_ids:
    # Instance ID 0 is reserved for crowd region.
    if instance_id == 0:
      continue
    thing_mask = tf.math.logical_and(instance_maps == instance_id,
                                     semantic_thing_segmentation)
    if tf.reduce_sum(tf.cast(thing_mask, tf.int32)) == 0:
      continue
    semantic_bin_counts = tf.math.bincount(
        tf.boolean_mask(semantic_prediction, thing_mask))
    semantic_majority = tf.cast(
        tf.math.argmax(semantic_bin_counts), tf.int32)

    while num_instance_per_semantic_label.size() <= semantic_majority:
      num_instance_per_semantic_label = num_instance_per_semantic_label.write(
          num_instance_per_semantic_label.size(), 0)

    new_instance_id = (
        num_instance_per_semantic_label.read(semantic_majority) + 1)
    num_instance_per_semantic_label = num_instance_per_semantic_label.write(
        semantic_majority, new_instance_id)
    panoptic_prediction = tf.where(
        thing_mask,
        tf.ones_like(panoptic_prediction) * semantic_majority * label_divisor
        + new_instance_id,
        panoptic_prediction)

  # Done with `num_instance_per_semantic_label` tensor array.
  num_instance_per_semantic_label.close()

  # Start to paste predicted `stuff` regions to panoptic prediction.
  instance_stuff_regions = instance_maps == 0
  semantic_ids, _ = tf.unique(tf.reshape(semantic_prediction, [-1]))
  for semantic_id in semantic_ids:
    if tf.reduce_sum(tf.cast(thing_class_ids == semantic_id, tf.int32)) > 0:
      continue
    # Check stuff area.
    stuff_mask = tf.math.logical_and(semantic_prediction == semantic_id,
                                     instance_stuff_regions)
    stuff_area = tf.reduce_sum(tf.cast(stuff_mask, tf.int32))
    if stuff_area >= stuff_area_limit:
      panoptic_prediction = tf.where(
          stuff_mask,
          tf.ones_like(panoptic_prediction) * semantic_id * label_divisor,
          panoptic_prediction)

  return panoptic_prediction
