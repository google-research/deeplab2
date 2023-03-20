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

"""Tensorflow code for working with object instances in segmentation."""

from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf


def instances_without_ignore_categories(panoptic_labels: tf.Tensor,
                                        ignore_categories: Union[tf.Tensor,
                                                                 Iterable[int]],
                                        panoptic_divisor: Union[tf.Tensor,
                                                                int] = 256):
  """Determines which instances to keep after ignoring a set of categories.

  Args:
    panoptic_labels: An integer tensor of panoptic labels of shape `[height,
      width]`. Each element will be `category * panoptic_divisor + instance` for
      a pixel.
    ignore_categories: An iterable or tensor of integer category labels.
      Instances where the category portion of the label in `panoptic_labels` are
      in the ignore set will not be included in the results.
    panoptic_divisor: The divisor used to multiply the category label when
      constructing panoptic labels, as in integer or integer scalar tensor.

  Returns:
    A boolean tensor masking which of the input `panoptic_labels` corresponds
    to an instance that will be kept, or equivalently *not* ignored.
  """
  ignore_categories = tf.convert_to_tensor(
      ignore_categories, dtype=panoptic_labels.dtype)
  panoptic_divisor = tf.convert_to_tensor(
      panoptic_divisor, dtype=panoptic_labels.dtype)

  instance_category = tf.math.floordiv(panoptic_labels, panoptic_divisor)
  instance_is_ignored = tf.math.reduce_any(
      tf.equal(
          tf.expand_dims(instance_category, 1),
          tf.expand_dims(ignore_categories, 0)),
      axis=1)
  instance_is_kept = tf.math.logical_not(instance_is_ignored)

  return instance_is_kept


def _broadcast_over_instances(t, num_instances):
  out_shape = tf.concat([tf.shape(t), [num_instances]], axis=0)
  return tf.broadcast_to(tf.expand_dims(t, -1), out_shape)


def instance_boxes_from_masks(
    panoptic_labels: tf.Tensor,
    ignore_categories: Optional[Union[tf.Tensor, Iterable[int]]] = None,
    panoptic_divisor: Union[tf.Tensor, int] = 256):
  """Finds the bounding boxes around instances, given a panoptic label map.

  Args:
    panoptic_labels: An integer tensor of panoptic labelsof shape `[height,
      width]`. Each element will be `category * panoptic_divisor + instance` for
      a pixel.
    ignore_categories: An iterable or tensor of integer category labels.
      Instances where the category portion of the label in `panoptic_labels` are
      in the ignore set will not be included in the results.
    panoptic_divisor: The divisor used to multiply the category label when
      constructing panoptic labels, as in integer or integer scalar tensor.

  Returns:
    A tuple of arrays (unique_labels, box_coords).
    unique_labels: An tensor of each possible non-ignored label value in
      `panoptic_labels`, in the same order as the boxes.
    box_coords: An tensor of shape `[num_labels, 4]`. Each row is one box as
      `[ymin, xmin, ymax, xmax]`.
  """
  label_shape = tf.shape(panoptic_labels)
  height = label_shape[0]
  width = label_shape[1]
  x_coord, y_coord = tf.meshgrid(
      tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))

  unique_labels, flat_instance_index = tf.unique(
      tf.reshape(panoptic_labels, [height * width]))
  num_instances = tf.size(unique_labels)
  instance_index = tf.reshape(flat_instance_index, [height, width])

  y_coord_repeated = _broadcast_over_instances(y_coord, num_instances)
  x_coord_repeated = _broadcast_over_instances(x_coord, num_instances)
  instance_index_repeated = _broadcast_over_instances(instance_index,
                                                      num_instances)

  instance_index_matches = tf.math.equal(
      instance_index_repeated,
      tf.reshape(tf.range(num_instances), [1, 1, num_instances]))

  # In these tensors, each slice in the 3rd dimension corresponds to an
  # instance. We replace the pixels that do _not_ belong to that instance with
  # a +/- infinity in order that it not be included in the reduce_min/max below.
  inf3d = tf.broadcast_to([[[float('Inf')]]], tf.shape(x_coord_repeated))
  y_or_inf = tf.where(instance_index_matches, y_coord_repeated, inf3d)
  y_or_neg_inf = tf.where(instance_index_matches, y_coord_repeated, -inf3d)
  x_or_inf = tf.where(instance_index_matches, x_coord_repeated, inf3d)
  x_or_neg_inf = tf.where(instance_index_matches, x_coord_repeated, -inf3d)

  y_min = tf.reduce_min(y_or_inf, axis=[0, 1])
  x_min = tf.reduce_min(x_or_inf, axis=[0, 1])
  y_max = tf.reduce_max(y_or_neg_inf, axis=[0, 1]) + 1
  x_max = tf.reduce_max(x_or_neg_inf, axis=[0, 1]) + 1

  box_coords = tf.stack([y_min, x_min, y_max, x_max], axis=1)

  if ignore_categories is not None:
    # Filter out the boxes that correspond to instances in the "ignore"
    # categories.
    instance_is_kept = instances_without_ignore_categories(
        unique_labels, ignore_categories, panoptic_divisor)

    unique_labels = tf.boolean_mask(unique_labels, instance_is_kept)
    box_coords = tf.boolean_mask(box_coords, instance_is_kept)

  return unique_labels, box_coords


def per_instance_masks(panoptic_labels: tf.Tensor,
                       instance_panoptic_labels: tf.Tensor,
                       out_dtype: tf.dtypes.DType = tf.bool) -> tf.Tensor:
  """3D tensor where each slice in 3rd dimensions is an instance's mask."""
  num_instances = tf.size(instance_panoptic_labels)
  matches = tf.equal(
      tf.expand_dims(panoptic_labels, 0),
      tf.reshape(instance_panoptic_labels, [num_instances, 1, 1]))
  return tf.cast(matches, out_dtype)


def _average_per_instance(map_tensor: tf.Tensor, panoptic_labels: tf.Tensor,
                          instance_panoptic_labels: tf.Tensor,
                          instance_area: tf.Tensor) -> tf.Tensor:
  """Finds the average of the values in map_tensor over each instance."""

  # For each instance (in the 3rd dim), generate a map that has, per-pixel:
  # - The input value if that pixel belongs to the instance.
  # - Zero otherwise.
  pixel_in_instance = per_instance_masks(panoptic_labels,
                                         instance_panoptic_labels)

  map_dtype = map_tensor.dtype
  num_instances = tf.size(instance_panoptic_labels)
  map_or_zero = tf.where(pixel_in_instance, tf.expand_dims(map_tensor, 0),
                         tf.zeros([num_instances, 1, 1], dtype=map_dtype))

  # Average the semantic probabilities over each instance.
  instance_total_prob = tf.math.reduce_sum(map_or_zero, axis=[1, 2])
  instance_avg_prob = tf.divide(instance_total_prob,
                                tf.cast(instance_area, map_dtype))

  return instance_avg_prob


# pyformat: disable
def per_instance_semantic_probabilities(
    panoptic_labels: tf.Tensor,
    instance_panoptic_labels: tf.Tensor,
    instance_area: tf.Tensor,
    semantic_probability: tf.Tensor,
    panoptic_divisor: Union[tf.Tensor, int],
    ignore_label: Union[tf.Tensor, int]) -> tf.Tensor:
  """Mean probability for the semantic label of each unique instance."""
  # pyformat: enable
  # Get the probability associated with the semantic label encoded in the
  # panoptic_labels at each pixel.
  panoptic_divisor = tf.convert_to_tensor(panoptic_divisor, dtype=tf.int32)
  ignore_label = tf.convert_to_tensor(ignore_label, dtype=tf.int32)
  semantic_label_map = tf.math.floordiv(panoptic_labels, panoptic_divisor)

  map_shape = tf.shape(semantic_label_map)
  height = map_shape[0]
  width = map_shape[1]
  num_pixels = height * width

  semantic_index = tf.reshape(semantic_label_map, [num_pixels])
  # Use 0 as the index for a pixel with the "ignore" label, since that semantic
  # label may not be a valid index into the class axis of the
  # semantic_probability tensor.
  semantic_index = tf.where(semantic_index == ignore_label, 0, semantic_index)

  x, y = tf.meshgrid(tf.range(width), tf.range(height))
  probability_index = tf.stack([
      tf.reshape(y, [num_pixels]),
      tf.reshape(x, [num_pixels]),
      semantic_index,
  ],
                               axis=1)

  # When len(semantic_probability.shape) == 3 (e.g., for Panoptic-DeepLab), it
  # is a semantic probability map with shape [H, W, num_classes], so we need to
  # gather the class confidence for corresponding semantic class prediction.
  # When len(semantic_probability.shape) == 2 (e.g., for MaX-DeepLab), it is a
  # semantic confidence map with shape [H, W], so we can directly use it as
  # pixel_semantic_probability.
  if len(semantic_probability.shape) == 3:
    pixel_semantic_probability = tf.reshape(
        tf.gather_nd(semantic_probability, probability_index), [height, width])
  elif len(semantic_probability.shape) == 2:
    pixel_semantic_probability = semantic_probability
  # Set the probability for the "ignore" pixels to 0.
  pixel_semantic_probability = tf.where(semantic_label_map == ignore_label, 0.0,
                                        pixel_semantic_probability)

  instance_avg_prob = _average_per_instance(pixel_semantic_probability,
                                            panoptic_labels,
                                            instance_panoptic_labels,
                                            instance_area)

  return instance_avg_prob


def combined_instance_scores(
    panoptic_labels: tf.Tensor, semantic_probability: tf.Tensor,
    instance_score_map: tf.Tensor, panoptic_divisor: Union[tf.Tensor, int],
    ignore_label: Union[tf.Tensor, int]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Combines (with a product) predicted semantic and instance probabilities.

  Args:
    panoptic_labels: A 2D integer tensor of panoptic format labels (each pixel
      entry is `semantic_label * panoptic_divisor + instance_label`).
    semantic_probability: A 3D float tensor, where the 3rd dimension is over
      semantic labels, and each spatial location will have the discrete
      distribution of the probabilities of the semantic classes.
    instance_score_map: A 2D float tensor, where the pixels for an instance will
      have the probability of that being an instance.
    panoptic_divisor: Integer scalar divisor/multiplier used to construct the
      panoptic labels.
    ignore_label: Integer scalar, for the "ignore" semantic label in the
      panoptic labels.

  Returns:
    A tuple of instance labels and the combined scores for those instances, each
    as a 1D tensor.
  """
  panoptic_divisor = tf.convert_to_tensor(panoptic_divisor, dtype=tf.int32)
  ignore_label = tf.convert_to_tensor(ignore_label, dtype=tf.int32)

  num_pixels = tf.size(panoptic_labels)
  instance_panoptic_labels, _, instance_area = tf.unique_with_counts(
      tf.reshape(panoptic_labels, [num_pixels]))

  instance_semantic_labels = tf.math.floordiv(instance_panoptic_labels,
                                              panoptic_divisor)
  valid_mask = tf.not_equal(instance_semantic_labels, ignore_label)
  instance_panoptic_labels = tf.boolean_mask(instance_panoptic_labels,
                                             valid_mask)
  instance_area = tf.boolean_mask(instance_area, valid_mask)

  instance_semantic_probabilities = per_instance_semantic_probabilities(
      panoptic_labels, instance_panoptic_labels, instance_area,
      semantic_probability, panoptic_divisor, ignore_label)

  instance_scores = _average_per_instance(instance_score_map, panoptic_labels,
                                          instance_panoptic_labels,
                                          instance_area)

  combined_scores = instance_semantic_probabilities * instance_scores
  return instance_panoptic_labels, combined_scores


def per_instance_is_crowd(is_crowd_map: tf.Tensor, id_map: tf.Tensor,
                          output_ids: tf.Tensor) -> tf.Tensor:
  """Determines the per-instance is_crowd value from a boolian is_crowd map.

  Args:
    is_crowd_map: A 2D boolean tensor. Where it is True, the instance in that
      region is a "crowd" instance. It is assumed that all pixels in an instance
      will have the same value in this map.
    id_map: A 2D integer tensor, with the instance id label at each pixel.
    output_ids: A 1D integer vector tensor, the per-instance ids for which to
      output the is_crowd values.

  Returns:
    A 1D boolean vector tensor, with the per-instance is_crowd value. The ith
    element of the return value will be the is_crowd result for the segment
    with the ith element of the output_ids argument.
  """
  flat_is_crowd_map = tf.reshape(is_crowd_map, [-1])
  flat_id_map = tf.reshape(id_map, [-1])

  # Get an is_crowd value from the map for each id.
  # Only need an arbtitrary value due to assumption that the is_crowd map does
  # not vary over an instance.
  unique_ids, unique_index = tf.unique(flat_id_map)
  unique_is_crowd = tf.scatter_nd(
      tf.expand_dims(unique_index, 1), flat_is_crowd_map, tf.shape(unique_ids))

  # Map from the order/set in unique_ids to that in output_ids
  matching_ids = tf.math.equal(
      tf.expand_dims(output_ids, 1), tf.expand_dims(unique_ids, 0))
  matching_index = tf.where(matching_ids)[:, 1]
  return tf.gather(unique_is_crowd, matching_index)
