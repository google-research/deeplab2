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

"""This file contains functions to post-process MaX-DeepLab results."""

import functools
from typing import List, Tuple, Dict, Text

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import utils


def _get_transformer_class_prediction(
    transformer_class_probs: tf.Tensor,
    transformer_class_confidence_threshold: float
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the transformer class prediction and confidence score.

  Args:
    transformer_class_probs: A tf.Tensor of shape [num_mask_slots,
      num_thing_stuff_classes + 1]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes and
      stuff classes) in MaX-DeepLab. The last channel indicates a `void` class.
    transformer_class_confidence_threshold: A float for thresholding the
      confidence of the transformer_class_probs. The panoptic mask slots with
      class confidence less than the threshold are filtered and not used for
      panoptic prediction. Only masks whose confidence is larger than the
      threshold are counted in num_detections.

  Returns:
    A tuple of:
    - the detected mask class prediction as float32 tf.Tensor of shape
      [num_detections].
    - the detected mask indices as tf.Tensor of shape [num_detections].
    - the number of detections as tf.Tensor of shape [1].
  """
  transformer_class_pred = tf.cast(
      tf.argmax(transformer_class_probs, axis=-1), tf.float32)
  transformer_class_confidence = tf.reduce_max(
      transformer_class_probs, axis=-1, keepdims=False)
  # Filter mask IDs with class confidence less than the threshold.
  thresholded_mask = tf.cast(
      tf.greater_equal(transformer_class_confidence,
                       transformer_class_confidence_threshold), tf.float32)
  transformer_class_confidence = (transformer_class_confidence
                                  * thresholded_mask)

  detected_mask_indices = tf.where(tf.greater(thresholded_mask, 0.5))[:, 0]
  detected_mask_class_pred = tf.gather(
      transformer_class_pred, detected_mask_indices)
  num_detections = tf.shape(detected_mask_indices)[0]
  return detected_mask_class_pred, detected_mask_indices, num_detections


def _get_mask_id_and_semantic_maps(
    thing_class_ids: List[int],
    stuff_class_ids: List[int],
    pixel_space_mask_logits: tf.Tensor,
    transformer_class_probs: tf.Tensor,
    image_shape: List[int],
    pixel_confidence_threshold=0.4,
    transformer_class_confidence_threshold=0.7,
    pieces=1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the pixel-level mask ID map and semantic map per image.

  Args:
    thing_class_ids: A List of integers of shape [num_thing_classes] containing
      thing class indices.
    stuff_class_ids: A List of integers of shape [num_thing_classes] containing
      stuff class indices.
    pixel_space_mask_logits: A tf.Tensor of shape [height, width,
      num_mask_slots]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes
      and stuff classes) in MaX-DeepLab.
    transformer_class_probs: A tf.Tensor of shape [num_mask_slots,
      num_thing_stuff_classes + 1]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes and
      stuff classes) in MaX-DeepLab. The last channel indicates a `void` class.
    image_shape: A list of integers specifying the [height, width] of input
      image.
    pixel_confidence_threshold: A float indicating a threshold for the pixel
      level softmax probability confidence of transformer mask logits. If less
      than the threshold, the pixel locations have confidence `0` in
      `confident_regions` output, and represent `void` (ignore) regions.
    transformer_class_confidence_threshold: A float for thresholding the
      confidence of the transformer_class_probs. The panoptic mask slots with
      class confidence less than the threshold are filtered and not used for
      panoptic prediction.
    pieces: An integer indicating the number of pieces in the piece-wise
      operation. When computing panpotic prediction and confident regions, the
      mask logits are divided width-wise into multiple pieces and processed
      piece-wise due to the GPU memory limit. Then, the piece-wise outputs are
      concatenated along the width into the original mask shape. Defaults to 1.

  Returns:
    A tuple of:
    - the mask ID prediction as tf.Tensor with shape [height, width].
    - the semantic prediction as tf.Tensor with shape [height, width].
    - the thing region mask as tf.Tensor with shape [height, width].
    - the stuff region mask as tf.Tensor with shape [height, width].

  Raises:
    ValueError: When input image's `width - 1` is not divisible by `pieces`.
  """
  # The last channel indicates `void` class and thus is not included.
  transformer_class_probs = transformer_class_probs[..., :-1]
  # Generate mapping from mask IDs to dataset's thing and stuff semantic IDs.
  thing_stuff_class_ids = thing_class_ids + stuff_class_ids

  detected_mask_class_pred, detected_mask_indices, num_detections = (
      _get_transformer_class_prediction(transformer_class_probs,
                                        transformer_class_confidence_threshold))
  # If num_detections = 0, return empty result maps.
  def _return_empty_mask_id_and_semantic_maps():
    return (
        tf.ones([image_shape[0], image_shape[1]], dtype=tf.int32),
        tf.zeros([image_shape[0], image_shape[1]], dtype=tf.int32),
        tf.zeros([image_shape[0], image_shape[1]], dtype=tf.float32),
        tf.zeros([image_shape[0], image_shape[1]], dtype=tf.float32))

  # If num_detections > 0:
  def _generate_mask_id_and_semantic_maps():
    output_mask_id_map = []
    output_confident_region = []
    logits_width = pixel_space_mask_logits.get_shape().as_list()[1]
    output_width = image_shape[1]

    if (output_width - 1) % pieces > 0:
      raise ValueError('`output_width - 1` must be divisible by `pieces`.')
    # Use of input shape of a multiple of the feature stride, plus one, so that
    # it preserves left- and right-alignment.
    piece_output_width = (output_width - 1) // pieces + 1

    for piece_id in range(pieces):
      piece_begin = (logits_width - 1) // pieces * piece_id
      # Use of input shape of a multiple of the feature stride, plus one, so
      # that it preserves left- and right-alignment.
      piece_end = (logits_width - 1) // pieces * (piece_id + 1) + 1
      piece_pixel_mask_logits = (
          pixel_space_mask_logits[:, piece_begin:piece_end, :])
      piece_pixel_mask_logits = tf.compat.v1.image.resize_bilinear(
          tf.expand_dims(piece_pixel_mask_logits, 0),
          (image_shape[0], piece_output_width),
          align_corners=True)
      piece_pixel_mask_logits = tf.squeeze(piece_pixel_mask_logits, axis=0)
      piece_detected_pixel_mask_logits = tf.gather(
          piece_pixel_mask_logits, detected_mask_indices, axis=-1)
      # Filter the pixels which are assigned to a mask ID that does not survive.
      piece_max_logits = tf.reduce_max(piece_pixel_mask_logits, axis=-1)
      piece_detected_max_logits = tf.reduce_max(
          piece_detected_pixel_mask_logits, axis=-1)
      piece_detected_mask = tf.cast(tf.math.equal(
          piece_max_logits, piece_detected_max_logits), tf.float32)
      # Filter with pixel mask threshold.
      piece_pixel_confidence_map = tf.reduce_max(
          tf.nn.softmax(piece_detected_pixel_mask_logits, axis=-1), axis=-1)
      piece_confident_region = tf.cast(
          piece_pixel_confidence_map > pixel_confidence_threshold, tf.float32)
      piece_confident_region = piece_confident_region * piece_detected_mask
      piece_mask_id_map = tf.cast(
          tf.argmax(piece_detected_pixel_mask_logits, axis=-1), tf.int32)
      if piece_id == pieces - 1:
        output_mask_id_map.append(piece_mask_id_map)
        output_confident_region.append(piece_confident_region)
      else:
        output_mask_id_map.append(piece_mask_id_map[:, :-1])
        output_confident_region.append(piece_confident_region[:, :-1])

    mask_id_map = tf.concat(output_mask_id_map, axis=1)
    confident_region = tf.concat(output_confident_region, axis=1)
    mask_id_map_flat = tf.reshape(mask_id_map, [-1])
    mask_id_semantic_map_flat = tf.gather(
        detected_mask_class_pred, mask_id_map_flat)
    mask_id_semantic_map = tf.reshape(
        mask_id_semantic_map_flat, [image_shape[0], image_shape[1]])
    # Generate thing and stuff masks (with value 1/0 indicates the
    # presence/absence)
    thing_mask = tf.cast(mask_id_semantic_map < len(thing_class_ids),
                         tf.float32) * confident_region
    stuff_mask = tf.cast(mask_id_semantic_map >= len(thing_class_ids),
                         tf.float32) * confident_region
    # Generate semantic_map.
    semantic_map = tf.gather(
        tf.convert_to_tensor(thing_stuff_class_ids),
        tf.cast(tf.round(mask_id_semantic_map_flat), tf.int32))
    semantic_map = tf.reshape(semantic_map, [image_shape[0], image_shape[1]])
    # Add 1 because mask ID 0 is reserved for unconfident region.
    mask_id_map_plus_one = mask_id_map + 1
    semantic_map = tf.cast(tf.round(semantic_map), tf.int32)
    return (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask)

  mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask = tf.cond(
      tf.cast(num_detections, tf.float32) < tf.cast(0.5, tf.float32),
      _return_empty_mask_id_and_semantic_maps,
      _generate_mask_id_and_semantic_maps)

  return (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask)


def _filter_by_count(input_index_map: tf.Tensor,
                     area_limit: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Filters input index map by area limit threshold per index.

  Args:
    input_index_map: A float32 tf.Tensor of shape [batch, height, width].
    area_limit: An integer specifying the number of pixels that each index
      regions need to have at least. If not over the limit, the index regions
      are masked (zeroed) out.

  Returns:
    masked input_index_map: A tf.Tensor with shape [batch, height, width],
      masked by the area_limit threshold.
    mask: A tf.Tensor with shape [batch, height, width]. It is a pixel-level
      mask with 1. indicating the regions over the area limit, and 0. otherwise.
  """
  batch_size = tf.shape(input_index_map)[0]
  index_map = tf.cast(tf.round(input_index_map), tf.int32)
  index_map_flat = tf.reshape(index_map, [batch_size, -1])
  counts = tf.math.bincount(index_map_flat, axis=-1)
  counts_map = tf.gather(counts, index_map_flat, batch_dims=1)
  counts_map = tf.reshape(counts_map, tf.shape(index_map))

  mask = tf.cast(
      tf.cast(counts_map, tf.float32) > tf.cast(area_limit - 0.5, tf.float32),
      input_index_map.dtype)
  return input_index_map * mask, mask


def _merge_mask_id_and_semantic_maps(
    mask_id_maps_plus_one: tf.Tensor,
    semantic_maps: tf.Tensor,
    thing_masks: tf.Tensor,
    stuff_masks: tf.Tensor,
    void_label: int,
    label_divisor: int,
    thing_area_limit: int,
    stuff_area_limit: int,) -> tf.Tensor:
  """Merges mask_id maps and semantic_maps to obtain panoptic segmentation.

  Args:
    mask_id_maps_plus_one: A tf.Tensor of shape [batch, height, width].
    semantic_maps: A tf.Tensor of shape [batch, height, width].
    thing_masks: A float32 tf.Tensor of shape [batch, height, width] containing
      masks with 1. at thing regions, 0. otherwise.
    stuff_masks: A float32 tf.Tensor of shape [batch, height, width] containing
      masks with 1. at thing regions, 0. otherwise.
    void_label: An integer specifying the void label.
    label_divisor: An integer specifying the label divisor of the dataset.
    thing_area_limit: An integer specifying the number of pixels that thing
      regions need to have at least. The thing region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least. The stuff region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.

  Returns:
    panoptic_maps: A tf.Tensor with shape [batch, height, width].

  """
  thing_mask_id_maps_plus_one = (tf.cast(mask_id_maps_plus_one, tf.float32)
                                 * thing_masks)
  # We increase semantic_maps by 1 before masking (zeroing) by thing_masks and
  # stuff_masks, to ensure all valid semantic IDs are greater than 0 and thus
  # not masked out.
  semantic_maps_plus_one = semantic_maps + 1
  tf.debugging.assert_less(
      tf.reduce_sum(thing_masks * stuff_masks), 0.5,
      message='thing_masks and stuff_masks must be mutually exclusive.')

  thing_semantic_maps = (tf.cast(semantic_maps_plus_one, tf.float32)
                         * thing_masks)
  stuff_semantic_maps = (tf.cast(semantic_maps_plus_one, tf.float32)
                         * stuff_masks)

  # Filter stuff_semantic_maps by stuff_area_limit.
  stuff_semantic_maps, _ = _filter_by_count(
      stuff_semantic_maps, stuff_area_limit)
  # Filter thing_mask_id_map and thing_semantic_map by thing_area_limit
  thing_mask_id_maps_plus_one, mask_id_count_filter_mask = _filter_by_count(
      thing_mask_id_maps_plus_one, thing_area_limit)
  thing_semantic_maps = thing_semantic_maps * mask_id_count_filter_mask

  # Filtered un-confident region will be replaced with `void_label`. The
  # "plus_one" will be reverted, the un-confident region (0) will be -1, and so
  # we add (void + 1)
  semantic_maps_new = thing_semantic_maps + stuff_semantic_maps - 1.0
  semantic_maps_new = (tf.cast(semantic_maps_new < -0.5, tf.float32)
                       * tf.cast(void_label + 1, tf.float32)
                       + semantic_maps_new)
  panoptic_maps = (semantic_maps_new * label_divisor
                   + thing_mask_id_maps_plus_one)
  panoptic_maps = tf.cast(tf.round(panoptic_maps), tf.int32)
  return panoptic_maps


def _get_panoptic_predictions(
    pixel_space_mask_logits: tf.Tensor,
    transformer_class_logits: tf.Tensor,
    thing_class_ids: List[int],
    void_label: int,
    label_divisor: int,
    thing_area_limit: int,
    stuff_area_limit: int,
    image_shape: List[int],
    pixel_confidence_threshold=0.4,
    transformer_class_confidence_threshold=0.7,
    pieces=1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the pixel-level panoptic, mask ID, and semantic maps.

  Args:
    pixel_space_mask_logits: A tf.Tensor of shape [batch, strided_height,
      strided_width, num_mask_slots]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes
      and stuff classes) in MaX-DeepLab.
    transformer_class_logits: A tf.Tensor of shape [batch, num_mask_slots,
      num_thing_stuff_classes + 1]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes and
      stuff classes) in MaX-DeepLab. The last channel indicates a `void` class.
    thing_class_ids: A List of integers of shape [num_thing_classes] containing
      thing class indices.
    void_label: An integer specifying the void label.
    label_divisor: An integer specifying the label divisor of the dataset.
    thing_area_limit: An integer specifying the number of pixels that thing
      regions need to have at least. The thing region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least. The stuff region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.
    image_shape: A list of integers specifying the [height, width] of input
      image.
    pixel_confidence_threshold: A float indicating a threshold for the pixel
      level softmax probability confidence of transformer mask logits. If less
      than the threshold, the pixel locations have confidence `0` in
      `confident_regions` output, and represent `void` (ignore) regions.
    transformer_class_confidence_threshold: A float for thresholding the
      confidence of the transformer_class_probs. The panoptic mask slots with
      class confidence less than the threshold are filtered and not used for
      panoptic prediction.
    pieces: An integer indicating the number of pieces in the piece-wise
      operation in `_get_mask_id_and_semantic_maps`. When computing panoptic
      prediction and confident regions, the mask logits are divided width-wise
      into multiple pieces and processed piece-wise due to the GPU memory limit.
      Then, the piece-wise outputs are concatenated along the width into the
      original mask shape. Defaults to 1.

  Returns:
    A tuple of:
    - the panoptic prediction as tf.Tensor with shape [batch, height, width].
    - the mask ID prediction as tf.Tensor with shape [batch, height, width].
    - the semantic prediction as tf.Tensor with shape [batch, height, width].
  """
  transformer_class_probs = tf.nn.softmax(transformer_class_logits, axis=-1)
  batch_size = tf.shape(transformer_class_logits)[0]
  # num_thing_stuff_classes does not include `void` class, so we decrease by 1.
  num_thing_stuff_classes = (
      transformer_class_logits.get_shape().as_list()[-1] - 1)
  # Generate thing and stuff class ids
  stuff_class_ids = utils.get_stuff_class_ids(
      num_thing_stuff_classes, thing_class_ids, void_label)

  mask_id_map_plus_one_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  semantic_map_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  thing_mask_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  stuff_mask_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  for i in tf.range(batch_size):
    mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask = (
        _get_mask_id_and_semantic_maps(
            thing_class_ids, stuff_class_ids,
            pixel_space_mask_logits[i, ...], transformer_class_probs[i, ...],
            image_shape, pixel_confidence_threshold,
            transformer_class_confidence_threshold, pieces)
        )
    mask_id_map_plus_one_lists = mask_id_map_plus_one_lists.write(
        i, mask_id_map_plus_one)
    semantic_map_lists = semantic_map_lists.write(i, semantic_map)
    thing_mask_lists = thing_mask_lists.write(i, thing_mask)
    stuff_mask_lists = stuff_mask_lists.write(i, stuff_mask)
  # This does not work with unknown shapes.
  mask_id_maps_plus_one = mask_id_map_plus_one_lists.stack()
  semantic_maps = semantic_map_lists.stack()
  thing_masks = thing_mask_lists.stack()
  stuff_masks = stuff_mask_lists.stack()

  panoptic_maps = _merge_mask_id_and_semantic_maps(
      mask_id_maps_plus_one, semantic_maps, thing_masks, stuff_masks,
      void_label, label_divisor, thing_area_limit, stuff_area_limit)
  return panoptic_maps, mask_id_maps_plus_one, semantic_maps


class PostProcessor(tf.keras.layers.Layer):
  """This class contains code of a MaX-DeepLab post-processor."""

  def __init__(
      self,
      config: config_pb2.ExperimentOptions,
      dataset_descriptor: dataset.DatasetDescriptor):
    """Initializes a MaX-DeepLab post-processor.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.
    """
    super(PostProcessor, self).__init__(name='PostProcessor')
    self._post_processor = functools.partial(
        _get_panoptic_predictions,
        thing_class_ids=list(dataset_descriptor.class_has_instances_list),
        void_label=dataset_descriptor.ignore_label,
        label_divisor=dataset_descriptor.panoptic_label_divisor,
        thing_area_limit=config.evaluator_options.thing_area_limit,
        stuff_area_limit=config.evaluator_options.stuff_area_limit,
        image_shape=list(config.eval_dataset_options.crop_size),
        transformer_class_confidence_threshold=config.evaluator_options
        .transformer_class_confidence_threshold,
        pixel_confidence_threshold=config.evaluator_options
        .pixel_confidence_threshold,
        pieces=1)

  def call(self, result_dict: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the post-processing given model predicted results.

    Args:
      result_dict: A dictionary of tf.Tensor containing model results. The dict
      has to contain
       - common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY,
       - common.PRED_TRANSFORMER_CLASS_LOGITS_KEY,

    Returns:
      The post-processed dict of tf.Tensor, containing the following:
       - common.PRED_SEMANTIC_KEY,
       - common.PRED_INSTANCE_KEY,
       - common.PRED_PANOPTIC_KEY,
    """
    processed_dict = {}
    (processed_dict[common.PRED_PANOPTIC_KEY],
     processed_dict[common.PRED_INSTANCE_KEY],
     processed_dict[common.PRED_SEMANTIC_KEY]
    ) = self._post_processor(
        result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY],
        result_dict[common.PRED_TRANSFORMER_CLASS_LOGITS_KEY])
    return processed_dict
