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

"""This file contains functions to post-process MaX-DeepLab results."""

import functools
from typing import List, Tuple, Dict, Text

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import utils

_SMALL_CONSTANT_FOR_SOFTMAX = -99999


def _get_transformer_class_prediction_thing_stuff(
    transformer_class_probs: tf.Tensor,
    transformer_class_confidence_threshold_thing: float,
    transformer_class_confidence_threshold_stuff: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the transformer class prediction and confidence score.

  Args:
    transformer_class_probs: A tf.Tensor of shape [num_mask_slots,
      num_thing_stuff_classes + 1]. It is a pixel level logit scores where the
      num_mask_slots is the number of mask slots (for both thing classes and
      stuff classes) in MaX-DeepLab. The last channel indicates a `void` class.
    transformer_class_confidence_threshold_thing: A float for thresholding the
      confidence of the transformer_class_probs. The panoptic mask slots with
      class confidence less than the threshold are filtered and not used for
      panoptic prediction. Only masks whose confidence is larger than the
      threshold are counted in num_detections. This threshold applies to thing
      classes.
    transformer_class_confidence_threshold_stuff: A float for thresholding the
      confidence of the transformer_class_probs. The panoptic mask slots with
      class confidence less than the threshold are filtered and not used for
      panoptic prediction. Only masks whose confidence is larger than the
      threshold are counted in num_detections. This threshold applies to stuff
      classes.

  Returns:
    A tuple of:
    - the class prediction for all mask slots as float32 tf.Tensor of shape
      [num_mask_slots].
    - the class prediction confidence as float32 tf.Tensor of shape
      [num_mask_slots].
    - the number of detections as tf.Tensor of shape [1].
  """
  transformer_class_pred = tf.cast(
      tf.argmax(transformer_class_probs, axis=-1), tf.float32)
  transformer_class_confidence = tf.reduce_max(
      transformer_class_probs, axis=-1, keepdims=False)
  # Filter mask IDs with class confidence less than the threshold.
  thresholded_mask_thing = tf.cast(
      tf.greater_equal(transformer_class_confidence,
                       transformer_class_confidence_threshold_thing),
      tf.float32)
  thresholded_mask_stuff = tf.cast(
      tf.greater_equal(transformer_class_confidence,
                       transformer_class_confidence_threshold_stuff),
      tf.float32)
  num_detections = tf.reduce_sum(
      tf.cast(
          tf.greater(thresholded_mask_stuff + thresholded_mask_thing, 0.5),
          tf.int32))

  # Instead of gathering detected mask slots and return, we simply return all
  # class prediction with a binary mask for keep/drop to ensure static shape for
  # TPU running.
  return transformer_class_pred, transformer_class_confidence, num_detections


def _get_transformer_class_prediction(
    transformer_class_probs: tf.Tensor,
    transformer_class_confidence_threshold: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    - the class prediction for all mask slots as float32 tf.Tensor of shape
      [num_mask_slots].
    - the binary indicator of detected mask as tf.Tensor of shape
      [num_mask_slots].
    - the class prediction confidence as float32 tf.Tensor of shape
      [num_mask_slots].
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
  num_detections = tf.reduce_sum(
      tf.cast(tf.greater(thresholded_mask, 0.5), tf.int32))

  # Instead of gathering detected mask slots and return, we simply return all
  # class prediction with a binary mask for keep/drop to ensure static shape for
  # TPU running.
  return (transformer_class_pred, thresholded_mask,
          transformer_class_confidence, num_detections)


def _get_mask_id_and_semantic_maps(
    thing_class_ids: List[int],
    stuff_class_ids: List[int],
    pixel_space_mask_logits: tf.Tensor,
    transformer_class_probs: tf.Tensor,
    image_shape: List[int],
    pixel_confidence_threshold=0.4,
    transformer_class_confidence_threshold=0.7,
    transformer_post_processing='pixelwise',
    maskwise_postprocessing_config=None,
    pieces=1
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the pixel-level mask ID map and semantic map per image or video.

   Some input and output tensors may contain num_frames in the shape, which is
   default to 1 (image-level processing).

  Args:
    thing_class_ids: A List of integers of shape [num_thing_classes] containing
      thing class indices.
    stuff_class_ids: A List of integers of shape [num_stuff_classes] containing
      stuff class indices.
    pixel_space_mask_logits: A tf.Tensor of shape [num_frames, height, width,
      num_mask_slots]. It is a pixel level logit scores where the num_mask_slots
      is the number of mask slots (for both thing classes and stuff classes) in
      MaX-DeepLab. num_frames is set to 1 by default (image-level processing).
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
    transformer_post_processing: A string indicating which post-processing is
      used to obtain panoptic prediction. Currently two types of
      post-processings are supported: 'pixelwise' and 'maskwise'.
    maskwise_postprocessing_config: A dictionary of hyper-parameters for
      maskwise postprocessing.
    pieces: An integer indicating the number of pieces in the piece-wise
      operation. When computing panpotic prediction and confident regions, the
      mask logits are divided width-wise into multiple pieces and processed
      piece-wise due to the GPU memory limit. Then, the piece-wise outputs are
      concatenated along the width into the original mask shape. Defaults to 1.

  Returns:
    A tuple of:
    - the mask ID prediction as tf.Tensor of shape [num_frames, height, width].
    - the semantic prediction as tf.Tensor of shape [num_frames, height, width].
    - the thing region mask as tf.Tensor of shape [num_frames, height, width].
    - the stuff region mask as tf.Tensor of shape [num_frames, height, width].
    - the instance score map as tf.Tensor of shape [num_frames, height, width].
    - the semantic score map as tf.Tensor of shape [num_frames, height, width].

  Raises:
    ValueError: When input image's `width - 1` is not divisible by `pieces`.
    ValueError: When transformer_post_processing is not one of ['pixelwise',
      'maskwise'].
  """
  # The last channel indicates `void` class and thus is not included.
  transformer_class_probs = transformer_class_probs[..., :-1]
  # Generate mapping from mask IDs to dataset's thing and stuff semantic IDs.
  thing_stuff_class_ids = thing_class_ids + stuff_class_ids

  if transformer_post_processing == 'pixelwise':
    (transformer_class_pred, transformer_class_thresholded_mask,
     transformer_class_confidence, num_detections) = (
         _get_transformer_class_prediction(
             transformer_class_probs, transformer_class_confidence_threshold))
  elif transformer_post_processing == 'maskwise':
    transformer_class_pred, transformer_class_confidence, num_detections = (
        _get_transformer_class_prediction_thing_stuff(
            transformer_class_probs, maskwise_postprocessing_config[
                'transformer_class_confidence_threshold_thing'],
            maskwise_postprocessing_config[
                'transformer_class_confidence_threshold_stuff']))
  else:
    raise ValueError('transformer_post_processing must be one of [pixelwise,',
                     ' maskwise].')
  tf.assert_rank(
      pixel_space_mask_logits, 4,
      'pixel_space_mask_logits must have shape [num_frames,'
      ' height, width, num_mask_slots] and rank 4.')

  if transformer_post_processing == 'maskwise':
    # We obtain a mask confidence as average of probs of all assigned pixels.
    num_mask_slots = pixel_space_mask_logits.get_shape().as_list()[-1]
    pixel_space_mask_probs = tf.nn.softmax(pixel_space_mask_logits, axis=-1)
    detected_pixel_mask = tf.cast(
        pixel_space_mask_probs > pixel_confidence_threshold, tf.float32)
    mask_scores_flat = tf.reshape(pixel_space_mask_probs, [-1, num_mask_slots])
    detected_pixel_mask_flat = tf.reshape(detected_pixel_mask,
                                          [-1, num_mask_slots])
    mask_scores_flat = tf.reduce_sum(
        mask_scores_flat * detected_pixel_mask_flat, axis=0)
    mask_scores_flat = tf.math.divide_no_nan(
        mask_scores_flat, tf.reduce_sum(detected_pixel_mask_flat, axis=0))

    reorder_score = (
        transformer_class_confidence **
        maskwise_postprocessing_config['reorder_class_weight']
    ) * (
        mask_scores_flat **
        maskwise_postprocessing_config['reorder_mask_weight'])

    mask_slots_reorder = tf.argsort(
        reorder_score, axis=-1, direction='DESCENDING', stable=True)
    # We re-order the predictions accordingly so they are arranged from more
    # confident to less.
    transformer_class_pred = tf.gather(
        transformer_class_pred, mask_slots_reorder, batch_dims=-1)
    transformer_class_confidence = tf.gather(
        transformer_class_confidence, mask_slots_reorder, batch_dims=-1)

  num_frames = pixel_space_mask_logits.get_shape().as_list()[0]
  # If num_detections = 0, return empty result maps.
  def _return_empty_mask_id_and_semantic_maps():
    return (tf.ones([num_frames, image_shape[0], image_shape[1]],
                    dtype=tf.int32),
            tf.zeros([num_frames, image_shape[0], image_shape[1]],
                     dtype=tf.int32),
            tf.zeros([num_frames, image_shape[0], image_shape[1]],
                     dtype=tf.float32),
            tf.zeros([num_frames, image_shape[0], image_shape[1]],
                     dtype=tf.float32),
            tf.zeros([num_frames, image_shape[0], image_shape[1]],
                     dtype=tf.float32),
            tf.zeros([num_frames, image_shape[0], image_shape[1]],
                     dtype=tf.float32))

  # If num_detections > 0 and transformer_post_processing == 'pixelwise':
  def _pixelwise_generate_mask_id_and_confident_maps():
    output_mask_id_map = []
    output_confident_region = []
    output_instance_score = []
    output_semantic_score = []
    logits_width = pixel_space_mask_logits.get_shape().as_list()[-2]
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
          pixel_space_mask_logits[..., piece_begin:piece_end, :])
      piece_pixel_mask_logits = tf.compat.v1.image.resize_bilinear(
          piece_pixel_mask_logits, (image_shape[0], piece_output_width),
          align_corners=True)
      # Set logits of undetected mask slots to a very small number, which
      # will set the corresponding value after softmax to 0 thus has no impact.
      piece_detected_pixel_mask_logits = (
          piece_pixel_mask_logits * transformer_class_thresholded_mask +
          (_SMALL_CONSTANT_FOR_SOFTMAX *
           (1 - transformer_class_thresholded_mask)))
      # Filter with pixel mask threshold.
      piece_pixel_confidence_map = tf.reduce_max(
          tf.nn.softmax(piece_detected_pixel_mask_logits, axis=-1), axis=-1)
      piece_confident_region = tf.cast(
          piece_pixel_confidence_map > pixel_confidence_threshold, tf.float32)
      # Filter the pixels which are assigned to a mask ID that does not survive.
      piece_max_logits = tf.reduce_max(piece_pixel_mask_logits, axis=-1)
      piece_detected_max_logits = tf.reduce_max(
          piece_detected_pixel_mask_logits, axis=-1)
      piece_detected_mask = tf.cast(
          tf.math.equal(piece_max_logits, piece_detected_max_logits),
          tf.float32)
      piece_confident_region = piece_confident_region * piece_detected_mask
      piece_mask_id_map = tf.cast(
          tf.argmax(piece_detected_pixel_mask_logits, axis=-1), tf.int32)
      piece_semantic_score = tf.gather(transformer_class_confidence,
                                       piece_mask_id_map)

      if piece_id == pieces - 1:
        output_mask_id_map.append(piece_mask_id_map)
        output_confident_region.append(piece_confident_region)
        output_instance_score.append(piece_pixel_confidence_map)
        output_semantic_score.append(piece_semantic_score)
      else:
        output_mask_id_map.append(piece_mask_id_map[..., :-1])
        output_confident_region.append(piece_confident_region[..., :-1])
        output_instance_score.append(piece_pixel_confidence_map[..., :-1])
        output_semantic_score.append(piece_semantic_score[..., :-1])
    return (output_mask_id_map, output_confident_region,
            output_instance_score, output_semantic_score)

  # If num_detections > 0 and transformer_post_processing == 'maskwise':
  def _maskwise_generate_mask_id_and_confident_maps():
    output_mask_id_map = []
    output_confident_region = []
    output_instance_score = []
    output_semantic_score = []
    logits_width = pixel_space_mask_logits.get_shape().as_list()[-2]
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
          pixel_space_mask_logits[..., piece_begin:piece_end, :])
      piece_pixel_mask_logits = tf.compat.v1.image.resize_bilinear(
          piece_pixel_mask_logits, (image_shape[0], piece_output_width),
          align_corners=True)

      piece_pixel_mask_probs = tf.nn.softmax(piece_pixel_mask_logits, axis=-1)
      # In mask-wise post-processing, we firstly obtain a set of binary masks
      # where overlapping is allowed.
      piece_detected_pixel_mask = tf.cast(
          piece_pixel_mask_probs > pixel_confidence_threshold, tf.float32)
      # Similarly, the binary masks need to be re-ordered.
      piece_detected_pixel_mask = tf.gather(
          piece_detected_pixel_mask, mask_slots_reorder, batch_dims=-1)

      # We firstly generate an empty canvas, and paste the predicted masks onto
      # it in order.
      piece_mask_id_map = tf.zeros(
          [num_frames, image_shape[0], piece_output_width], dtype=tf.float32)
      piece_confident_region = tf.zeros(
          [num_frames, image_shape[0], piece_output_width], dtype=tf.float32)
      piece_instance_score = tf.reduce_max(piece_pixel_mask_probs, axis=-1)
      piece_semantic_score = tf.zeros(
          [num_frames, image_shape[0], piece_output_width], dtype=tf.float32)

      for mask_slots_idx in range(num_mask_slots):
        current_binary_mask = piece_detected_pixel_mask[..., mask_slots_idx]
        current_class_confidence = transformer_class_confidence[mask_slots_idx]
        current_class_pred = transformer_class_pred[mask_slots_idx]
        # We need to tell if current class prediction belongs to thing class or
        # stuff class, as different threshold may be used.
        is_current_class_pred_thing = tf.cast(
            current_class_pred < len(thing_class_ids), tf.float32)
        is_current_class_pred_confident = (
            tf.cast(
                current_class_confidence >= maskwise_postprocessing_config[
                    'transformer_class_confidence_threshold_thing'], tf.float32)
            * is_current_class_pred_thing + tf.cast(
                current_class_confidence >= maskwise_postprocessing_config[
                    'transformer_class_confidence_threshold_stuff'],
                tf.float32) * (1 - is_current_class_pred_thing))
        # When copying current binary mask onto the canvas, some pixels may
        # already be occuipied by other masks with higher confidence, thus we
        # should remove these pixels from current binary mask. However, if too
        # many pixels are removed, we may also consider remove the whole mask.
        # maskwise_merge_overlap_threshold is used to decide when a whole mask
        # should be removed at this case.
        original_pixel_number = tf.reduce_sum(current_binary_mask)
        # We can only copy mask onto those unoccupied pixels.
        new_binary_mask = current_binary_mask * (
            tf.cast(piece_mask_id_map == 0, tf.float32))
        new_pixel_number = tf.reduce_sum(new_binary_mask)
        current_mask_not_overlap_too_much = tf.cast(
            new_pixel_number > original_pixel_number *
            maskwise_postprocessing_config['overlapping_threshold'], tf.float32)
        # We keep new_binary_mask if it is confident and do not overlap too
        # much with previous masks.
        new_binary_mask = (
            new_binary_mask * is_current_class_pred_confident *
            current_mask_not_overlap_too_much)
        # Paste the mask onto canvas.
        piece_mask_id_map = (
            piece_mask_id_map +
            new_binary_mask * tf.cast(mask_slots_idx + 1, tf.float32))
        piece_confident_region = (piece_confident_region + new_binary_mask)
        piece_semantic_score = ((1.0 - new_binary_mask) * piece_semantic_score
                                + new_binary_mask * current_class_confidence)

      # At final, we minus one so the mask_id will correctly start from 0. Note
      # that the unconfident pixels may be wrongly classified to mask slot 0,
      # but it is fine as it will finally be filtered by confident_region_mask.
      # In pixelwise post-processing, the piece_mask_id_map will range from 0
      # to N-1, and unconfident pixels will still have the mask_id based on
      # argmax, which is filtered based on confident_region variable afterwards.
      # Here as we use maskwise post-processing, we do not know unconfident
      # pixels' "correct" assginment based on argmax, thus we simply assgin
      # them all to mask slot 0, which is filtered based on confident_region
      # variable afterwards. This can also ensure that piece_mask_id_map will
      # range from 0 to N-1.
      piece_mask_id_map = piece_mask_id_map - 1
      piece_mask_id_map = tf.maximum(piece_mask_id_map, 0)
      piece_mask_id_map = tf.cast(piece_mask_id_map, tf.int32)

      if piece_id == pieces - 1:
        output_mask_id_map.append(piece_mask_id_map)
        output_confident_region.append(piece_confident_region)
        output_instance_score.append(piece_instance_score)
        output_semantic_score.append(piece_semantic_score)
      else:
        output_mask_id_map.append(piece_mask_id_map[..., :-1])
        output_confident_region.append(piece_confident_region[..., :-1])
        output_instance_score.append(piece_instance_score[..., :-1])
        output_semantic_score.append(piece_semantic_score[..., :-1])
    return (output_mask_id_map, output_confident_region,
            output_instance_score, output_semantic_score)

  def _generate_mask_id_and_semantic_maps():
    (output_mask_id_map, output_confident_region, output_instance_score,
     output_semantic_score) = (
         _pixelwise_generate_mask_id_and_confident_maps()
         if transformer_post_processing == 'pixelwise' else
         _maskwise_generate_mask_id_and_confident_maps())
    mask_id_map = tf.concat(output_mask_id_map, axis=-1)
    confident_region = tf.concat(output_confident_region, axis=-1)
    instance_score_map = tf.concat(output_instance_score, axis=-1)
    semantic_score_map = tf.concat(output_semantic_score, axis=-1)
    mask_id_map_flat = tf.reshape(mask_id_map, [-1])
    mask_id_semantic_map_flat = tf.gather(transformer_class_pred,
                                          mask_id_map_flat)
    mask_id_semantic_map = tf.reshape(
        mask_id_semantic_map_flat, [num_frames, image_shape[0], image_shape[1]])

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
    semantic_map = tf.reshape(semantic_map,
                              [num_frames, image_shape[0], image_shape[1]])
    # Add 1 because mask ID 0 is reserved for unconfident region.
    mask_id_map_plus_one = mask_id_map + 1
    semantic_map = tf.cast(tf.round(semantic_map), tf.int32)
    return (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask,
            instance_score_map, semantic_score_map)

  (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask,
   instance_score_map, semantic_score_map) = tf.cond(
       tf.cast(num_detections, tf.float32) < tf.cast(0.5, tf.float32),
       _return_empty_mask_id_and_semantic_maps,
       _generate_mask_id_and_semantic_maps)

  return (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask,
          instance_score_map, semantic_score_map)


def _filter_by_count(input_index_map: tf.Tensor,
                     area_limit: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Filters input index map by area limit threshold per index.

  Args:
    input_index_map: A float32 tf.Tensor of shape [batch, num_frames, height,
      width].
    area_limit: An integer specifying the number of pixels that each index
      regions need to have at least. If not over the limit, the index regions
      are masked (zeroed) out.

  Returns:
    masked input_index_map: A tf.Tensor with shape [batch, num_frames, height,
      width], masked by the area_limit threshold.
    mask: A tf.Tensor with shape [batch, num_frames, height, width]. It is a
      pixel-level mask with 1. indicating the regions over the area limit, and
      0. otherwise.
  """
  input_h, input_w = input_index_map.get_shape().as_list()[-2:]
  index_map = tf.cast(tf.round(input_index_map), tf.int32)
  index_map_flat = tf.reshape(index_map, [-1, input_h * input_w])
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
    stuff_area_limit: int,
) -> tf.Tensor:
  """Merges mask_id maps and semantic_maps to obtain panoptic segmentation.

  Args:
    mask_id_maps_plus_one: A tf.Tensor of shape [batch, height, width] or
      [batch, num_frames, height, width].
    semantic_maps: A tf.Tensor of shape [batch, height, width] or [batch,
      num_frames, height, width].
    thing_masks: A float32 tf.Tensor of shape [batch, height, width] or [batch,
      num_frames, height, width] containing masks with 1. at thing regions, 0.
      otherwise.
    stuff_masks: A float32 tf.Tensor of shape [batch, height, width] or [batch,
      num_frames, height, width] containing masks with 1. at thing regions, 0.
      otherwise.
    void_label: An integer specifying the void label.
    label_divisor: An integer specifying the label divisor of the dataset.
    thing_area_limit: An integer specifying the number of pixels that thing
      regions need to have at least. The thing region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label. A function to filter out thing
      regions smaller than thing_area_limit is applied when thing_area_limit is
      larger than 0.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least. The stuff region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label. A function to filter out stuff
      regions smaller than stuff_area_limit is applied when stuff_area_limit is
      larger than 0.

  Returns:
    panoptic_maps: A tf.Tensor with shape [batch, num_frames, height, width].
  """
  thing_mask_id_maps_plus_one = (
      tf.cast(mask_id_maps_plus_one, tf.float32) * thing_masks)
  # We increase semantic_maps by 1 before masking (zeroing) by thing_masks and
  # stuff_masks, to ensure all valid semantic IDs are greater than 0 and thus
  # not masked out.
  semantic_maps_plus_one = semantic_maps + 1
  tf.debugging.assert_less(
      tf.reduce_sum(thing_masks * stuff_masks),
      0.5,
      message='thing_masks and stuff_masks must be mutually exclusive.')

  thing_semantic_maps = (
      tf.cast(semantic_maps_plus_one, tf.float32) * thing_masks)
  stuff_semantic_maps = (
      tf.cast(semantic_maps_plus_one, tf.float32) * stuff_masks)
  if stuff_area_limit > 0:
    # Filter stuff_semantic_maps by stuff_area_limit.
    stuff_semantic_maps, _ = _filter_by_count(stuff_semantic_maps,
                                              stuff_area_limit)
  if thing_area_limit > 0:
    # Filter thing_mask_id_map and thing_semantic_map by thing_area_limit
    thing_mask_id_maps_plus_one, mask_id_count_filter_mask = _filter_by_count(
        thing_mask_id_maps_plus_one, thing_area_limit)
    thing_semantic_maps = thing_semantic_maps * mask_id_count_filter_mask

  # Filtered un-confident region will be replaced with `void_label`. The
  # "plus_one" will be reverted, the un-confident region (0) will be -1, and so
  # we add (void + 1)
  semantic_maps_new = thing_semantic_maps + stuff_semantic_maps - 1.0
  semantic_maps_new = (
      tf.cast(semantic_maps_new < -0.5, tf.float32) *
      tf.cast(void_label + 1, tf.float32) + semantic_maps_new)
  panoptic_maps = (
      semantic_maps_new * label_divisor + thing_mask_id_maps_plus_one)
  panoptic_maps = tf.cast(tf.round(panoptic_maps), tf.int32)
  return panoptic_maps


@tf.function
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
    transformer_post_processing='pixelwise',
    maskwise_postprocessing_config=None,
    pieces=1
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the pixel-level panoptic, mask ID, and semantic maps.

  Args:
    pixel_space_mask_logits: A tf.Tensor of shape [batch, num_frames,
      strided_height, strided_width, num_mask_slots]. It is a pixel level logit
      scores where the num_mask_slots is the number of mask slots (for both
      thing classes and stuff classes) in MaX-DeepLab.
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
    transformer_post_processing: A string indicating which post-processing is
      used to obtain panoptic prediction. Currently two types of
      post-processings are supported: 'pixelwise' and 'maskwise'.
    maskwise_postprocessing_config: A dictionary of hyper-parameters for
      maskwise postprocessing.
    pieces: An integer indicating the number of pieces in the piece-wise
      operation in `_get_mask_id_and_semantic_maps`. When computing panoptic
      prediction and confident regions, the mask logits are divided width-wise
      into multiple pieces and processed piece-wise due to the GPU memory limit.
      Then, the piece-wise outputs are concatenated along the width into the
      original mask shape. Defaults to 1.

  Returns:
    A tuple of:
    - the panoptic prediction as tf.Tensor with shape [batch, num_frames,
        height, width].
    - the mask ID prediction as tf.Tensor with shape [batch, num_frames, height,
        width].
    - the semantic prediction as tf.Tensor with shape [batch, num_frames,
        height, width].
    - the instance score map as tf.Tensor with shape [batch, num_frames,
        height, width].
    - the semantic score map as tf.Tensor with shape [batch, num_frames,
        height, width].
  """
  transformer_class_probs = tf.nn.softmax(transformer_class_logits, axis=-1)
  batch_size = utils.resolve_batch_size(transformer_class_logits)
  # num_thing_stuff_classes does not include `void` class, so we decrease by 1.
  num_thing_stuff_classes = (
      transformer_class_logits.get_shape().as_list()[-1] - 1)
  # Generate thing and stuff class ids
  stuff_class_ids = utils.get_stuff_class_ids(num_thing_stuff_classes,
                                              thing_class_ids, void_label)

  mask_id_map_plus_one_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  semantic_map_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  thing_mask_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  stuff_mask_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  instance_score_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  semantic_score_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  for i in tf.range(batch_size):
    (mask_id_map_plus_one, semantic_map, thing_mask, stuff_mask,
     instance_score_map, semantic_score_map) = (
         _get_mask_id_and_semantic_maps(thing_class_ids, stuff_class_ids,
                                        pixel_space_mask_logits[i, ...],
                                        transformer_class_probs[i, ...],
                                        image_shape, pixel_confidence_threshold,
                                        transformer_class_confidence_threshold,
                                        transformer_post_processing,
                                        maskwise_postprocessing_config, pieces))
    mask_id_map_plus_one_lists = mask_id_map_plus_one_lists.write(
        i, mask_id_map_plus_one)
    semantic_map_lists = semantic_map_lists.write(i, semantic_map)
    thing_mask_lists = thing_mask_lists.write(i, thing_mask)
    stuff_mask_lists = stuff_mask_lists.write(i, stuff_mask)
    instance_score_map_lists = (
        instance_score_map_lists.write(i, instance_score_map))
    semantic_score_map_lists = (
        semantic_score_map_lists.write(i, semantic_score_map))
  # This does not work with unknown shapes.
  mask_id_maps_plus_one = mask_id_map_plus_one_lists.stack()
  semantic_maps = semantic_map_lists.stack()
  thing_masks = thing_mask_lists.stack()
  stuff_masks = stuff_mask_lists.stack()
  instance_score_maps = instance_score_map_lists.stack()
  semantic_score_maps = semantic_score_map_lists.stack()

  panoptic_maps = _merge_mask_id_and_semantic_maps(
      mask_id_maps_plus_one, semantic_maps, thing_masks, stuff_masks,
      void_label, label_divisor, thing_area_limit, stuff_area_limit)
  return (panoptic_maps, mask_id_maps_plus_one, semantic_maps,
          instance_score_maps, semantic_score_maps)


class PostProcessor(tf.keras.layers.Layer):
  """This class contains code of a MaX-DeepLab post-processor."""

  def __init__(self,
               config: config_pb2.ExperimentOptions,
               dataset_descriptor: dataset.DatasetDescriptor,
               name: str = 'PostProcessor'):
    """Initializes a MaX-DeepLab post-processor.

    We currently support two different ways for post-processing:
      - pixel-wise [1]: In pixel-wise post-processing, the panoptic prediction
        is obtained with two argmax, where the first argmax decicdes the
        assignment from pixels to mask slots, and second one decides the
        predicted semantic class of each mask slot.

      - mask-wise [2,3,4]: In mask-wise post-processing, the panoptic prediction
        is obtained in a copy-paste manner. Specifically, a set of binary masks
        is obtained for each mask slot by thresholding. Afterwards, the binary
        masks are re-ordered with a confidence score based on class
        confidence and/or mask confidence. Then the masks are copied to an
        empty canvas in the order from high confidence to low confidence.

    References:
      [1] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
          CVPR 2021.
            Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

      [2] Per-Pixel Classification is Not All You Need for Semantic
          Segmentation, NeurIPS 2021.
            Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.

      [3] Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with
          Transformers, CVPR 2022.
            Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar,
            Jose M. Alvarez, Ping Luo, Tong Lu.

      [4] CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation,
          CVPR 2022.
            Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins,
            Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.
      name: Name of the layer.
    """
    super(PostProcessor, self).__init__(name=name)
    # Convert maskwise_postprocessing_config to a dict.
    maskwise_postprocessing_config = None
    if config.evaluator_options.HasField('maskwise_postprocessing'):
      maskwise_postprocessing_config = {
          'transformer_class_confidence_threshold_thing':
              config.evaluator_options.maskwise_postprocessing
              .transformer_class_confidence_threshold_thing,
          'transformer_class_confidence_threshold_stuff':
              config.evaluator_options.maskwise_postprocessing
              .transformer_class_confidence_threshold_stuff,
          'overlapping_threshold':
              config.evaluator_options.maskwise_postprocessing
              .overlapping_threshold,
          'reorder_class_weight':
              config.evaluator_options.maskwise_postprocessing
              .reorder_class_weight,
          'reorder_mask_weight':
              config.evaluator_options.maskwise_postprocessing
              .reorder_mask_weight,
      }

    self._post_processor = functools.partial(
        _get_panoptic_predictions,
        thing_class_ids=list(dataset_descriptor.class_has_instances_list),
        void_label=dataset_descriptor.ignore_label,
        label_divisor=dataset_descriptor.panoptic_label_divisor,
        thing_area_limit=config.evaluator_options.thing_area_limit,
        stuff_area_limit=config.evaluator_options.stuff_area_limit,
        image_shape=list(config.eval_dataset_options.crop_size),
        pixel_confidence_threshold=config.evaluator_options
        .pixel_confidence_threshold,
        transformer_class_confidence_threshold=config.evaluator_options
        .transformer_class_confidence_threshold,
        transformer_post_processing=config.evaluator_options
        .transformer_post_processing,
        maskwise_postprocessing_config=maskwise_postprocessing_config,
        pieces=1)

    # We use another post-processor for semantic prediction and evaluation,
    # where we do not apply any thresholding. When run panoptic model for
    # semantic segmentation, we set the threshold to 0 for a better results
    # (as semantic label does not prefer 'void'). With another post-processor
    # where no thresholding is applied, we do not need to run the model twice,
    # and we may have more accurate evaluation of mIoU on the fly.
    self._post_processor_semantic = functools.partial(
        _get_panoptic_predictions,
        thing_class_ids=list(dataset_descriptor.class_has_instances_list),
        void_label=dataset_descriptor.ignore_label,
        label_divisor=dataset_descriptor.panoptic_label_divisor,
        thing_area_limit=0,
        stuff_area_limit=0,
        image_shape=list(config.eval_dataset_options.crop_size),
        pixel_confidence_threshold=0,
        transformer_class_confidence_threshold=0,
        transformer_post_processing='pixelwise',
        maskwise_postprocessing_config=None,
        pieces=1)

    # Post-processing for video data is not fully supported yet, so we fix the
    # num_frames to 1.
    self._num_frames = 1

  def _expand_num_frames_dimension(self, result_dict):
    """Expands the num_frames dimension of pixel predictions."""
    pixel_space = result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY]
    pixel_shape = pixel_space.get_shape().as_list()
    # Expand the num_frames dimension as [batch * num_frames, height, width,
    # num_mask_slots] -> [batch, num_frames, height, width, num_mask_slots].
    if len(pixel_shape) == 4:
      result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY] = tf.reshape(
          pixel_space, [-1, self._num_frames] + pixel_shape[1:])
    return result_dict

  def _merge_batch_and_num_frames_dimensions(self, result_dict):
    """Merges the batch and num_frames dimensions of the pixel predictions."""
    for key in result_dict.keys():
      # Squeeze [batch, num_frames, height, width] -> [batch * num_frames,
      # height, width]
      result_shape = result_dict[key].get_shape().as_list()
      if len(result_shape) >= 4:
        result_dict[key] = tf.reshape(result_dict[key], [-1] + result_shape[2:])
    return result_dict

  def call(self, result_dict: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the post-processing given model predicted results.

    Args:
      result_dict: A dictionary of tf.Tensor containing model results. The dict
        has to contain:
          - common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY,
          - common.PRED_TRANSFORMER_CLASS_LOGITS_KEY,

    Returns:
      The post-processed dict of tf.Tensor, containing the following:
       - common.PRED_SEMANTIC_KEY,
       - common.PRED_INSTANCE_KEY,
       - common.PRED_PANOPTIC_KEY,
    """
    # Make a copy so that input argument will not be modified, per requirements
    # from exporting a saved model.
    result_dict = dict(result_dict)

    # We handle both image [batch, height, width, ...] and video [batch,
    # num_frames, height, width, ...] inputs. To unify the input and output
    # shape for all the post-processing functions, we expand the num_frames
    # dimension for image-level tensors (num_frames = 1) before calling the
    # post-process function. Then, we merge the batch and num_frames dimensions.
    result_dict = self._expand_num_frames_dimension(result_dict)
    processed_dict = {}
    (processed_dict[common.PRED_PANOPTIC_KEY],
     processed_dict[common.PRED_INSTANCE_KEY],
     _,
     processed_dict[common.PRED_INSTANCE_SCORES_KEY],
     processed_dict[common.PRED_SEMANTIC_SCORES_KEY]) = self._post_processor(
         result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY],
         result_dict[common.PRED_TRANSFORMER_CLASS_LOGITS_KEY])

    # We obtain semantic predictions with another post-processor, where
    # thresholding is not applied.
    (_,
     _,
     processed_dict[common.PRED_SEMANTIC_KEY],
     _,
     _) = self._post_processor_semantic(
         result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY],
         result_dict[common.PRED_TRANSFORMER_CLASS_LOGITS_KEY])

    processed_dict = self._merge_batch_and_num_frames_dimensions(processed_dict)

    return processed_dict
