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

"""This file contains the loss functions for MaX-DeepLab models.

Reference:
  MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
    CVPR 2021. https://arxiv.org/abs/2012.00759
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

  CMT-DeepLab: "Clustering Mask Transformers for Panoptic Segmentation",
    CVPR 2022. https://arxiv.org/abs/2206.08948
      Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""
from typing import Text, Dict, Tuple, List

import tensorflow as tf
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model import utils
from deeplab2.model.loss import base_loss
from deeplab2.model.loss import matchers_ops

# Positive and negative constants that are used to pad or mask hungarian
# matching weights.
_MATCHING_NEGATIVE_CONSTANT = -999.0
_MATCHING_POSITIVE_CONSTANT = 999.0
# A large negative constant applied before softmax. This will make the softmax
# ignore the masked logits.
_SOFTMAX_MASKING_CONSTANT = -99999.0

_GT_KEY = 'gt_key'
_PRED_KEY = 'pred_key'
_WEIGHT_KEY = 'weight_key'


def _gumbel_topk_sample(logits, k):
  """Samples k points from the softmax distribution with Gumbel-Top-k trick."""
  gumbel_noise = -tf.math.log(-tf.math.log(
      tf.random.uniform(tf.shape(logits), 0, 1)))
  _, indices = tf.nn.top_k(logits + gumbel_noise, k)
  return indices


def _generate_mask_slot_semantic_one_hot(
    matched_mask_slot_indices: tf.Tensor,
    mask_gt_semantic_map: tf.Tensor,
    num_mask_slots: int,
    thing_stuff_class_ids: List[int]):
  """Generates the ground truth for transformer_class_logits.

  This function generates a pseudo ground truth that we will use to train the
  transformer class head logits. The input tensors, matched_mask_slot_indices
  and mask_gt_semantic_map, are obtained by (hungarian) matching the ground
  truth masks with the predicted masks. Note that this function generates the
  positive one hot encodings only, i.e., the void class is not included in the
  output tensor but will be generated outside the function.

  Args:
    matched_mask_slot_indices: An int32 tf.Tensor of shape [batch_size,
      num_ground_truth_masks] that encodes the matched mask slot id for each
      ground truth mask.
    mask_gt_semantic_map: An int32 tf.Tensor of shape [batch_size,
      num_ground_truth_masks] that encodes the semantic label for each ground
      truth mask. A padded mask (or void, or no object) will have the label -1.
    num_mask_slots: An integer, the number of mask slots for the MaX-DeepLab
      model.
    thing_stuff_class_ids: A list of integers of length [num_thing_classes +
      num_stuff_classes] that encodes the class IDs for all thing and stuff
      classes. It is a concatenation of the thing_class_ids list and the
      stuff_class_ids list.

  Returns:
    mask_slot_semantic_one_hot: An output tf.Tensor with shape [batch_size,
      num_mask_slots, num_thing_classes + num_stuff_classes].
  """
  semantic_map_shape = mask_gt_semantic_map.get_shape().as_list()
  batch_size = utils.resolve_batch_size(mask_gt_semantic_map)
  num_ground_truth_masks = semantic_map_shape[-1]

  # Concatenate the indices in each dimension of the ground truth one hot
  # output.
  batch_indices = tf.expand_dims(tf.range(batch_size), axis=-1)
  batch_indices = tf.tile(batch_indices, [1, num_ground_truth_masks])
  batch_indices = tf.reshape(batch_indices, [-1, 1])
  matched_mask_slot_indices = tf.reshape(matched_mask_slot_indices, [-1, 1])
  # We shift the semantic map by one so that void labels (-1) will be a valid
  # index too. Otherwise, tf.scatter_nd raises error if it runs on CPU.
  semantic_indices = tf.reshape(mask_gt_semantic_map, [-1, 1]) + 1
  indices = tf.concat([batch_indices,
                       matched_mask_slot_indices,
                       semantic_indices], axis=-1)

  # Generate mask_slot_semantic_one_hot by scattering constant ones onto a
  # constant zero tensor.
  updates = tf.ones([batch_size * num_ground_truth_masks], dtype=tf.float32)
  mask_slot_semantic_one_hot = tf.tensor_scatter_nd_add(
      tf.zeros(
          [batch_size, num_mask_slots, max(thing_stuff_class_ids) + 2],
          updates.dtype), indices, updates)

  # Gather the wanted classes in the desired (thing + stuff) order.
  thing_stuff_tensor = tf.cast(thing_stuff_class_ids, tf.int32)
  # We also shift the thing_stuff_tensor index by one in order to revert the
  # semantic map shifting above.
  mask_slot_semantic_one_hot = tf.gather(mask_slot_semantic_one_hot,
                                         thing_stuff_tensor + 1, axis=2)
  return mask_slot_semantic_one_hot


def nonsquare_hungarian_matching(
    weights: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Hungarian matching with arbitrary shape.

  The matchers_ops.hungarian_matching supports only squared weight matrices.
  This function generalizes the hungarian matching to nonsquare cases by padding
  the weights to a square and running the square version matching. The property
  of hungarian matching ensures that the solutions are equivalent for the padded
  square problem and the original nonsquare problem.

  Args:
    weights: A [batch, shape1, shape2] float32 tf.Tensor.

  Returns:
    square_permutation: A [batch, max(shape1, shape2), max(shape1, shape2)]
      float32 tf.Tensor that is the permutation matrix that achieves the minimum
      total weight. Note that a permutation matrix contains only value 0.0 and
      1.0, with each row and each column sums to 1.0.
    nonsquare_permutation: A [batch, shape1, shape2] float32 tf.Tensor. The
      nonsquare part of the permutation matrix.
  """
  _, height, width = weights.get_shape().as_list()
  max_height_width = max(height, width)
  # Padding a constant on one axis does not affect matching results.
  weights = tf.pad(weights,
                   [[0, 0],  # Do not pad the batch dimension.
                    [0, max_height_width - height],
                    [0, max_height_width - width]],
                   constant_values=_MATCHING_NEGATIVE_CONSTANT)
  square_permutation = matchers_ops.hungarian_matching(weights)

  square_permutation = tf.cast(square_permutation, tf.float32)
  return square_permutation, square_permutation[:, :height, :width]


def _mask_similarity(gt_mask: tf.Tensor, pred_mask: tf.Tensor,
                     metric: str = 'dice') -> tf.Tensor:
  """Computes mask similarity between gt_masks and pred_masks.

  Args:
    gt_mask: A [batch, height * width, num_gt_masks] float32 tf.Tensor, that
      contains only value 0.0 and 1.0. Each 1.0 indicates that the pixel belongs
      to the ground truth mask. Note that panoptic segmentation enforces that
      ground truth masks do not overlap.
    pred_mask: A [batch, height * width, num_pred_masks] float32 tf.Tensor, that
      is positive. For each batch_id and pixel_id, the [num_pred_masks] vector
      encodes whether each pixel belongs to each mask. The sum of each vector is
      less than or equal to one.
    metric: A string, the mask similarity metric that we will compute. Supports
      'dice' (default), 'iou', 'intersection_over_ground_truth', and
      'intersection_over_prediction'.

  Returns:
    mask_similarity: A float32 [batch, num_gt_masks, num_pred_masks] tf.Tensor
      that contains the mask similarity between all ground truth masks and all
      predicted masks.

  Raises:
    ValueError: If the mask similarity metric is not one of 'dice', 'iou',
    'intersection_over_ground_truth', or 'intersection_over_prediction'.
  """
  denominator_epsilon = 1e-5
  intersection = tf.einsum('bpi,bpj->bij', gt_mask, pred_mask)
  if metric.lower() == 'dice':
    denominator = (tf.expand_dims(tf.reduce_sum(gt_mask, axis=1), axis=2) +
                   tf.reduce_sum(pred_mask, axis=1, keepdims=True)) / 2
  elif metric.lower() == 'iou':
    denominator = (tf.expand_dims(tf.reduce_sum(gt_mask, axis=1), axis=2) +
                   tf.reduce_sum(pred_mask, axis=1, keepdims=True) -
                   intersection)
  elif metric.lower() == 'intersection_over_ground_truth':
    denominator = tf.expand_dims(tf.reduce_sum(gt_mask, axis=1), axis=2)
  elif metric.lower() == 'intersection_over_prediction':
    denominator = tf.reduce_sum(pred_mask, axis=1, keepdims=True)
  else:
    raise ValueError('The mask similarity metric is not supported.')
  return intersection / (denominator + denominator_epsilon)


class MaXDeepLabLoss(tf.keras.layers.Layer):
  """This class contains code for MaX-DeepLab losses."""

  def __init__(self,
               loss_options: config_pb2.LossOptions,
               num_classes: int,
               ignore_label: int,
               thing_class_ids: Tuple[int],
               focal_loss_alpha: float = 0.75,
               instance_discrimination_sample_k: int = 0,
               instance_discrimination_sample_temperature: float = 0.6,
               instance_discrimination_temperature: float = 0.3,
               semantic_sample_k: int = 0,
               semantic_sample_temperature: float = 1.0,
               auxiliary_output_number: int = 0):
    """Initializes a MaX-DeepLab loss.

    This class supports PQ-style loss, mask id cross entropy loss, and instance
    discrimination loss, proposed in MaX-DeepLab. It also supports semantic loss
    as an auxiliary target. The PQ-style loss can be further decomposed in to a
    classification term and a mask dice term.

    Reference:
      MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

    Args:
      loss_options: Loss options as defined by config_pb2.LossOptions.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An integer specifying the ignore label.
      thing_class_ids: A tuple of length [N] containing N thing indices.
      focal_loss_alpha: An optional float specifying the coefficient that
        weights between positive (matched) and negative (unmatched) masks in
        focal loss. The positives are weighted by alpha, while the negatives
        are weighted by (1. - alpha). Note that we do not use a focal loss
        gamma here, i.e., the gamma is set to zero which is equivalent to the
        normal cross-entropy loss, except for the alpha weighting. Default to
        0.75.
      instance_discrimination_sample_k: An optional integer specifying the
        number of points or features that we will sample to perform the pixel
        contrastive instance discrimination loss. If this is set to 0, then we
        use the original MaX-DeepLab instance discrimination loss. Otherwise,
        we replace the original instance discrimination loss with a pixel
        contrastive instance discrimination loss. Default to 0 as in the
        original MaX-DeepLab.
      instance_discrimination_sample_temperature: An optional float temperature
        specifying the sample temperature of the points or features for the
        pixel contrastive instance discrimination loss. This is only used when
        instance_discrimination_sample_k > 0. Default to 0.6, which seems good
        on Cityscapes.
      instance_discrimination_temperature: An optional float specifying the
        temperature for the instance discrimination loss.
      semantic_sample_k: An optional integer specifying the
        number of points or features that we will sample to perform the semantic
        loss. If this is set to 0, then we use all pixels for semantic loss.
        Otherwise, we subsample pixels based on object scale for semantic loss.
        Default to 0.
      semantic_sample_temperature: An optional float temperature
        specifying the sample temperature of the points or features for the
        semantic loss. The sampling is based on object scale. When
        temperature > 1, the sampling will bias to smaller object, while a
        value < 1 will lead to a sampling bias to larger object. This is only
        used when semantic_sample_k > 0. Default to 1.0.
      auxiliary_output_number: An integer specifying the number of auxiliary
        outputs.
    """
    super(MaXDeepLabLoss, self).__init__(name='MaXDeepLabLoss')
    # The loss_terms will optionally include
    #  - common.PQ_STYLE_LOSS_CLASS_TERM
    #  - common.PQ_STYLE_LOSS_MASK_DICE_TERM
    #  - common.MASK_ID_CROSS_ENTROPY_LOSS
    #  - common.INSTANCE_DISCRIMINATION_LOSS
    #  - common.SEMANTIC_LOCSS

    self._auxiliary_output_number = auxiliary_output_number

    # These loss terms will be accessed by loss_builder.py and will be used to
    # build loss metrics.
    self.loss_terms = []

    # The PQ-style loss includes two terms.
    self._pq_style_loss_weight = 0.0
    if loss_options.HasField(common.PQ_STYLE_LOSS):
      self._pq_style_loss_weight = loss_options.pq_style_loss.weight
      self.loss_terms.append(common.PQ_STYLE_LOSS_CLASS_TERM)
      self.loss_terms.append(common.PQ_STYLE_LOSS_MASK_DICE_TERM)
      # Add PQ-style loss for auxiliary outputs.
      for auxiliary_idx in range(auxiliary_output_number):
        self.loss_terms.append('aux_' + str(auxiliary_idx) +
                               common.PQ_STYLE_LOSS_CLASS_TERM)
        self.loss_terms.append('aux_' + str(auxiliary_idx) +
                               common.PQ_STYLE_LOSS_MASK_DICE_TERM)

    # Mask-ID cross entropy loss. For semantic (stuff-only) segmentation, the
    # Mask-ID cross entropy loss works as a semantic loss.
    self._mask_id_cross_entropy_loss_weight = 0.0
    self._mask_id_cross_entropy_loss_topk = 1.0
    if loss_options.HasField(common.MASK_ID_CROSS_ENTROPY_LOSS):
      self._mask_id_cross_entropy_loss_weight = (
          loss_options.mask_id_cross_entropy_loss.weight)
      self._mask_id_cross_entropy_loss_topk = (
          loss_options.mask_id_cross_entropy_loss.top_k_percent)
      self.loss_terms.append(common.MASK_ID_CROSS_ENTROPY_LOSS)
      # Add Mask-ID cross entropy loss for auxiliary outputs.
      for auxiliary_idx in range(auxiliary_output_number):
        self.loss_terms.append('aux_' + str(auxiliary_idx) +
                               common.MASK_ID_CROSS_ENTROPY_LOSS)

    # Instance discrimination loss.
    self._instance_discrimination_sample_k = instance_discrimination_sample_k
    self._instance_discrimination_sample_temperature = (
        instance_discrimination_sample_temperature)
    # Set whether to use stuff regions for instance discrimination. Note this
    # only affects the original instance discrimination loss which by default
    # uses thing regions only (i.e., only when instance_discrimination_sample_k
    # = 0). The pixel contrastive loss uses both thing and stuff regions by
    # default.
    self._instance_discrimination_loss_weight = 0.0
    self._instance_discrimination_loss_topk = 1.0
    if loss_options.HasField(common.INSTANCE_DISCRIMINATION_LOSS):
      self._instance_discrimination_loss_weight = (
          loss_options.instance_discrimination_loss.weight)
      self._instance_discrimination_loss_topk = (
          loss_options.instance_discrimination_loss.top_k_percent)
      self.loss_terms.append(common.INSTANCE_DISCRIMINATION_LOSS)
      # Add Instance discrimination loss for auxiliary outputs.
      for auxiliary_idx in range(auxiliary_output_number):
        self.loss_terms.append('aux_' + str(auxiliary_idx) +
                               common.INSTANCE_DISCRIMINATION_LOSS)

    # Semantic loss.
    self._semantic_loss_weight = 0.0
    self._semantic_loss_topk = 1.0
    self._semantic_sample_k = semantic_sample_k
    self._semantic_sample_temperature = semantic_sample_temperature
    if loss_options.HasField(common.SEMANTIC_LOSS):
      self._semantic_loss_weight = loss_options.semantic_loss.weight
      self._semantic_loss_topk = loss_options.semantic_loss.top_k_percent
      self.loss_terms.append(common.SEMANTIC_LOSS)

    self._ignore_label = ignore_label
    self._thing_class_ids = list(thing_class_ids)
    self._focal_loss_alpha = focal_loss_alpha
    self._instance_discrimination_temperature = (
        instance_discrimination_temperature)

    # Build the base loss functions.
    self._pq_style_loss_class_term = base_loss.FocalCrossEntropyLoss(
        gt_key=_GT_KEY, pred_key=_PRED_KEY, weight_key=_WEIGHT_KEY,
        # Num_classes and ignore_label are not necessary since the inputs will
        # be one hot encoded already.
        num_classes=None, ignore_label=None,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=0.0, background_channel_index=-1,
        dynamic_weight=True)
    self._pq_style_loss_mask_dice_term = base_loss.MaskDiceLoss(
        gt_key=_GT_KEY, pred_key=_PRED_KEY, weight_key=_WEIGHT_KEY,
        prediction_activation='softmax')
    self._mask_id_cross_entropy_loss = base_loss.TopKCrossEntropyLoss(
        gt_key=_GT_KEY, pred_key=_PRED_KEY, weight_key=_WEIGHT_KEY,
        # Num_classes and ignore_label are not necessary since the inputs will
        # be one hot encoded already.
        num_classes=None, ignore_label=None,
        top_k_percent_pixels=self._mask_id_cross_entropy_loss_topk,
        dynamic_weight=True)
    self._instance_discrimination_loss = base_loss.TopKCrossEntropyLoss(
        gt_key=_GT_KEY, pred_key=_PRED_KEY, weight_key=_WEIGHT_KEY,
        # Num_classes and ignore_label are not necessary since the inputs will
        # be one hot encoded already.
        num_classes=None, ignore_label=None,
        top_k_percent_pixels=self._instance_discrimination_loss_topk,
        dynamic_weight=True)
    self._semantic_loss = base_loss.TopKCrossEntropyLoss(
        common.GT_SEMANTIC_KEY,
        common.PRED_SEMANTIC_LOGITS_KEY,
        common.SEMANTIC_LOSS_WEIGHT_KEY,
        num_classes=num_classes,
        ignore_label=ignore_label,
        top_k_percent_pixels=self._semantic_loss_topk)

  def build(self, input_shapes: Tuple[Dict[Text, tf.Tensor], Dict[Text,
                                                                  tf.Tensor]]):
    """Extracts useful constants that depend on the input shapes."""
    y_true_shapes = input_shapes[0]
    self._max_thing_id = int(y_true_shapes[common.GT_THING_ID_CLASS_KEY][-1])
    y_pred_shapes = input_shapes[1]
    transformer_class_logits_shape = y_pred_shapes[
        common.PRED_TRANSFORMER_CLASS_LOGITS_KEY]
    self._num_mask_slots = int(transformer_class_logits_shape[1])
    # The transformer_class_logits contain thing classes, stuff classes, and the
    # void class, so num_thing_stuff_classes should be the total number of
    # classes minus one.
    self._num_thing_stuff_classes = int(transformer_class_logits_shape[2]) - 1
    # Since we implement the PQ-style loss with the class term plus the mask
    # dice term (Equation 10 of the paper), we need to balance the two terms to
    # have the same weight and normalizating constants. The focal loss alpha is
    # a weight on the positive class term, so we apply it to the mask dice term
    # too. The class loss is also normalized by the number of mask slots, so we
    # do the same normalization for the mask dice term.
    self._mask_dice_term_modifier = (
        self._focal_loss_alpha / self._num_mask_slots)

    self._stuff_class_ids = utils.get_stuff_class_ids(
        self._num_thing_stuff_classes,
        self._thing_class_ids,
        self._ignore_label)
    self._num_stuff_classes = len(self._stuff_class_ids)
    self._thing_stuff_class_ids = self._thing_class_ids + self._stuff_class_ids
    self._pixel_gt_num_mask_id = self._max_thing_id + self._num_stuff_classes

  def _pre_process_ground_truth(
      self, y_true: Dict[Text, tf.Tensor], output_height: int, output_width: int
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
             tf.Tensor, tf.Tensor]:
    """Pre-processes the ground truth before we compute the losses.

    This function generates tensors that do not depend on the prediction of the
    model, but are useful to the calculation of the losses. The function mainly
    downsamples the pixel space ground truth to the model output resolution, and
    combines (or concatenates) the thing masks and the stuff masks. The output
    shape pixel_gt_num_mask_id = max_thing_id + num_stuff_classes, which means
    the output masks contain both thing masks and stuff masks.

    Args:
      y_true: A dict of tensors providing ground-truth information, containing
       - common.GT_SEMANTIC_KEY: A [batch, height, width] int32 tf.Tensor, the
         semantic label map.
       - common.GT_THING_ID_MASK_KEY: A [batch, height, width] int32 tf.Tensor.
         It assigns each non-crowd thing instance a unique mask-ID label,
         starting from 0. Unassigned pixels are set to -1.
       - common.GT_THING_ID_CLASS_KEY: A [batch, max_thing_id] int32 tf.Tensor.
         It contains semantic ID of each instance assigned to thing_id_mask. The
         remaining (max_thing_id - num_things) elements are set to -1.
      output_height: An integer, the height of the model output.
      output_width: An integer, the width of the model output.

    Returns:
      pixel_gt_thing_mask: A [batch, output_height * output_width] float32
        tensor, with value 0.0 and 1.0 only, indicating whether a pixel belongs
        to a 'thing' class.
      pixel_gt_non_void_mask: A [batch, output_height * output_width] float32
        tensor, with value 0.0 and 1.0 only, indicating if a pixel does not
        belong to the void class.
      pixel_gt_mask_id_one_hot: A [batch, output_height * output_width,
        pixel_gt_num_mask_id] float32 tensor, with value 0.0 and 1.0 only,
        indicating the mask id each pixel belongs to.
      mask_gt_semantic_map: A [batch, pixel_gt_num_mask_id] int32 tensor, the
        semantic class of each ground truth mask.
      mask_gt_non_void_mask: A [batch, pixel_gt_num_mask_id] int32 tensor, with
        value 0.0 and 1.0 only, indicating if the ground truth mask is a valid
        mask, not a padded mask. The masks are padded because TPU does not
        support dynamic shapes except in the batch axis. We pad all ground truth
        thing masks to a large enough constant max_thing_id. Similarly, stuff
        classes that do not present in the current image will be set to a void
        mask too.
      mask_gt_semantic_one_hot: A [batch, pixel_gt_num_mask_id,
        num_thing_stuff_classes] float32 tensor, with value 0.0 and 1.0 only,
        containing the one hot encodings of the ground truth mask classes. The
        last dimension contains concatenated thing classes and stuff classes,
        which is different from the dataset class IDs in mask_gt_semantic_map.
      mask_gt_area: A [batch, pixel_gt_num_mask_id] float32 tensor, the area of
        each ground truth mask. Padded masks have an area of 0.0.
      inverse_pixel_gt_mask_area: A [batch, output_height * output_width]
        float32 tensor, containing the inverse of ground truth mask area for
        each pixel. More specifically, the value equals
        output_height * output_width / pixel_gt_mask_area. The mask area is
        normalized by the total area (output_height * output_width) at the
        current output resolution, so that the mask areas are roughly invariant
        to the setting of different output strides.
    """
    # The depth of one hot encoding should be the largest id plus one. For
    # example, if we want to one-hot encode a class ID of 133 (the largest ID
    # for the COCO dataset), we will need a one-hot encoding of length 134.
    one_hot_depth = max(self._thing_stuff_class_ids) + 1
    batch_size = utils.resolve_batch_size(y_true[common.GT_SEMANTIC_KEY])

    # Compute pixel_gt_semantic_map (downsampling and reshaping to the 1D
    # representation that will be mainly used in this loss function).
    pixel_gt_semantic_map = utils.strided_downsample(
        y_true[common.GT_SEMANTIC_KEY],
        target_size=[output_height, output_width])

    pixel_gt_semantic_map = tf.reshape(
        pixel_gt_semantic_map,
        [batch_size, output_height * output_width])

    # Compute pixel_gt_non_void_mask.
    pixel_gt_non_void_mask = tf.cast(
        tf.not_equal(pixel_gt_semantic_map, self._ignore_label), tf.float32)
    pixel_gt_non_void_mask = tf.ensure_shape(
        pixel_gt_non_void_mask,
        [None, output_height * output_width])

    # Compute pixel_gt_semantic_one_hot from pixel_gt_semantic_map in order to
    # gather pixel_gt_stuff_id_one_hot from pixel_gt_semantic_one_hot.
    pixel_gt_semantic_one_hot = tf.one_hot(pixel_gt_semantic_map, one_hot_depth)
    # Convert the one hot encoding from the dataset id order to (thing, stuff)
    # order used in MaX-DeepLab.
    pixel_gt_stuff_id_one_hot = tf.gather(pixel_gt_semantic_one_hot,
                                          self._stuff_class_ids, axis=-1)
    pixel_gt_stuff_id_one_hot = tf.ensure_shape(
        pixel_gt_stuff_id_one_hot,
        [None, output_height * output_width, self._num_stuff_classes])

    # Compute pixel_gt_thing_id_one_hot for thing masks.
    pixel_gt_thing_id_map = utils.strided_downsample(
        y_true[common.GT_THING_ID_MASK_KEY],
        target_size=[output_height, output_width])
    pixel_gt_thing_id_map = tf.reshape(
        pixel_gt_thing_id_map, shape=[batch_size, output_height * output_width])
    # Note that common.GT_THING_ID_MASK_KEY uses -1 for void masks. And 0 to
    # (num_mask_slots - 1) are used for num_mask_slots mask slots.
    pixel_gt_thing_mask = tf.cast(
        tf.not_equal(pixel_gt_thing_id_map, -1), tf.float32)
    pixel_gt_thing_id_one_hot = tf.one_hot(pixel_gt_thing_id_map,
                                           self._max_thing_id)
    # Compute pixel_gt_mask_id_one_hot by concatenating thing masks with stuff
    # masks.
    pixel_gt_mask_id_one_hot = tf.concat([pixel_gt_thing_id_one_hot,
                                          pixel_gt_stuff_id_one_hot], axis=-1)
    pixel_gt_mask_id_one_hot = tf.ensure_shape(
        pixel_gt_mask_id_one_hot,
        [None, output_height * output_width, self._pixel_gt_num_mask_id])

    # Compute mask_gt_area by summing the one hot encodings spatially.
    mask_gt_area = tf.expand_dims(
        tf.reduce_sum(pixel_gt_mask_id_one_hot, axis=1), axis=-1)
    # Generate a binary mask for ground truth masks, indicating whether each
    # ground truth mask exists in the pixel space with a non-zero area. Note
    # that a mask that exists in the original input resolution will be removed
    # if its area is zero in the output resolution, due to downsampling.
    mask_gt_area_mask = tf.reshape(mask_gt_area > 0.5,
                                   [batch_size, self._pixel_gt_num_mask_id])

    # Compute inverse_pixel_gt_mask_area.
    pixel_gt_mask_area = tf.einsum('bpm,bmd->bp',
                                   pixel_gt_mask_id_one_hot, mask_gt_area)
    inverse_pixel_gt_mask_area = (output_height * output_width /
                                  tf.maximum(pixel_gt_mask_area, 1.0))

    # Compute mask_gt_semantic_map and mask_gt_semantic_one_hot.
    thing_id_gt_semantic_map = tf.reshape(
        tf.cast(y_true[common.GT_THING_ID_CLASS_KEY], tf.int32),
        [batch_size, self._max_thing_id])
    # The stuff ground truth semantic map is just the stuff class IDs.
    stuff_id_gt_semantic_map = tf.tile(
        tf.reshape(
            tf.cast(self._stuff_class_ids, tf.int32),
            [1, self._num_stuff_classes]), [batch_size, 1])
    mask_gt_semantic_map = tf.concat(
        [thing_id_gt_semantic_map, stuff_id_gt_semantic_map], axis=-1)
    # Set masks with zero area to void (-1), which is consistent with the void
    # label used in common.GT_THING_ID_CLASS_KEY but is different from the
    # ignore_labels of the datasets.
    mask_gt_semantic_map = (
        (mask_gt_semantic_map + 1) * tf.cast(mask_gt_area_mask, tf.int32) - 1)
    # Void (-1) classes will automatically be ignored by tf.one_hot.
    mask_gt_semantic_one_hot = tf.one_hot(mask_gt_semantic_map, one_hot_depth)
    mask_gt_semantic_one_hot = tf.gather(
        mask_gt_semantic_one_hot, self._thing_stuff_class_ids, axis=-1)

    # Compute mask_gt_non_void_mask. Again, a mask that exists in the original
    # input resolution is set to void if its area is zero in the output
    # resolution, due to downsampling.
    mask_gt_non_void_mask = tf.cast(mask_gt_semantic_map > -1, tf.float32)
    mask_gt_non_void_mask = tf.ensure_shape(
        mask_gt_non_void_mask, [None, self._pixel_gt_num_mask_id])

    return (pixel_gt_thing_mask, pixel_gt_non_void_mask,
            pixel_gt_mask_id_one_hot, mask_gt_semantic_map,
            mask_gt_non_void_mask, mask_gt_semantic_one_hot,
            mask_gt_area, inverse_pixel_gt_mask_area)

  def _call_pixel_contrastive_instance_discrimination_loss(
      self,
      resulting_dict: Dict[Text, tf.Tensor],
      pixel_feature: tf.Tensor,
      pixel_gt_mask_id_one_hot: tf.Tensor,
      inverse_pixel_gt_mask_area: tf.Tensor,
      pixel_gt_non_void_mask: tf.Tensor):
    """Applies the pixel contrastive instance discrimination loss [1].

    References:
      [1] CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation,
          CVPR 2022.
            Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins,
            Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

    Args:
      resulting_dict: The resulting loss as a dict of tf.Tensor.
      pixel_feature: A [batch, output_height * output_width, channel] float32
        tensor, the pixel space normalized feature.
      pixel_gt_mask_id_one_hot: A [batch, output_height * output_width,
        pixel_gt_num_mask_id] float32 tensor, with value 0.0 and 1.0 only,
        indicating the mask id each pixel belongs to.
      inverse_pixel_gt_mask_area: A [batch, output_height * output_width]
        float32 tensor, containing the inverse of ground truth mask area for
        each pixel. More specifically, the value equals
        output_height * output_width / pixel_gt_mask_area. The mask area is
        normalized by the total area (output_height * output_width) at the
        current output resolution, so that the mask areas are roughly invariant
        to the setting of different output strides.
      pixel_gt_non_void_mask: A [batch, output_height * output_width] float32
        tensor, with value 0.0 and 1.0 only, indicating if a pixel does not
        belong to the void class.
    """
    # Compute pixel space sampling indices according to mask area.
    sample_logits = (
        tf.math.log(inverse_pixel_gt_mask_area) *
        self._instance_discrimination_sample_temperature)
    sample_logits += (1 - pixel_gt_non_void_mask) * _SOFTMAX_MASKING_CONSTANT
    sample_indices = _gumbel_topk_sample(sample_logits,
                                         self._instance_discrimination_sample_k)

    # Sample ground truth one-hot encodings and compute gt_similarity.
    pixel_gt_sampled_feature = tf.gather(pixel_gt_mask_id_one_hot,
                                         sample_indices, batch_dims=1)
    sampled_gt_similarity = tf.einsum('bki,bji->bkj',
                                      pixel_gt_sampled_feature,
                                      pixel_gt_sampled_feature)

    # Normalize the ground truth similarity into a distribution (sum to 1).
    pixel_normalizing_constant = tf.reduce_sum(
        sampled_gt_similarity, axis=-1, keepdims=True)
    sampled_gt_similarity /= tf.maximum(pixel_normalizing_constant, 1.0)

    # Sample predicted features and compute pred_similarity.
    pixel_pred_sampled_feature = tf.gather(pixel_feature,
                                           sample_indices, batch_dims=1)
    sampled_pred_similarity = tf.einsum('bki,bji->bkj',
                                        pixel_pred_sampled_feature,
                                        pixel_pred_sampled_feature)
    sampled_pred_similarity /= (
        self._instance_discrimination_temperature)

    # Auxiliary instance_discrimination_loss.
    if self._instance_discrimination_loss_weight > 0.0:
      batch_size = utils.resolve_batch_size(pixel_feature)
      resulting_dict[common.INSTANCE_DISCRIMINATION_LOSS] = (
          self._instance_discrimination_loss(
              {_GT_KEY: sampled_gt_similarity},
              {_PRED_KEY: sampled_pred_similarity,
               _WEIGHT_KEY: tf.ones(
                   [batch_size, self._instance_discrimination_sample_k])}) *
          self._instance_discrimination_loss_weight)

  def _call_original_instance_discrimination_loss(
      self,
      resulting_dict: Dict[Text, tf.Tensor],
      pixel_feature: tf.Tensor,
      pixel_gt_mask_id_one_hot: tf.Tensor,
      mask_gt_area: tf.Tensor,
      mask_gt_non_void_mask: tf.Tensor,
      pixel_gt_thing_mask: tf.Tensor):
    """Applies the original MaX-DeepLab instance discrimination loss.

    Args:
      resulting_dict: The resulting loss as a dict of tf.Tensor.
      pixel_feature: A [batch, output_height * output_width, channel] float32
        tensor, the pixel space normalized feature.
      pixel_gt_mask_id_one_hot: A [batch, output_height * output_width,
        pixel_gt_num_mask_id] float32 tensor, with value 0.0 and 1.0 only,
        indicating the mask id each pixel belongs to.
      mask_gt_area: A [batch, pixel_gt_num_mask_id] float32 tensor, the area of
        each ground truth mask. Padded masks have an area of 0.0.
      mask_gt_non_void_mask: A [batch, pixel_gt_num_mask_id] int32 tensor, with
        value 0.0 and 1.0 only, indicating if the ground truth mask is a valid
        mask, not a padded mask. The masks are padded because TPU does not
        support dynamic shapes except in the batch axis. We pad all ground truth
        thing masks to a large enough constant max_thing_id. Similarly, stuff
        classes that do not present in the current image will be set to a void
        mask too.
      pixel_gt_thing_mask: A [batch, output_height * output_width] float32
        tensor, with value 0.0 and 1.0 only, indicating whether a pixel belongs
        to a 'thing' class.
    """
    # Compute mask_average_feature by averaging the feature of each mask.
    mask_average_feature = tf.einsum(
        'bpd,bpi->bid',
        pixel_feature,
        pixel_gt_mask_id_one_hot) / tf.maximum(mask_gt_area, 1.0)
    # Normalize the mask feature as the pixel space output feature is usually
    # normalized too.
    mask_average_feature = tf.math.l2_normalize(mask_average_feature, axis=-1)

    # Compute instance_discrimination_similarity, scaled by a constant
    # temperature.
    instance_discrimination_similarity = tf.einsum(
        'bpd,bid->bpi', pixel_feature, mask_average_feature)
    instance_discrimination_similarity /= (
        self._instance_discrimination_temperature)
    mask_gt_non_void_mask_expanded_1 = tf.expand_dims(
        mask_gt_non_void_mask, axis=1)
    # Mask void masks by setting them to a large negative value, so that they
    # will be ignored by the softmax in the loss.
    instance_discrimination_similarity = (
        mask_gt_non_void_mask_expanded_1 *
        instance_discrimination_similarity +
        (1.0 - mask_gt_non_void_mask_expanded_1) * _SOFTMAX_MASKING_CONSTANT)

    # Auxiliary instance_discrimination_loss.
    if self._instance_discrimination_loss_weight > 0.0:
      resulting_dict[common.INSTANCE_DISCRIMINATION_LOSS] = (
          self._instance_discrimination_loss(
              {_GT_KEY: pixel_gt_mask_id_one_hot},
              {_PRED_KEY: instance_discrimination_similarity,
               _WEIGHT_KEY: pixel_gt_thing_mask}) *
          self._instance_discrimination_loss_weight)

  def _call_semantic_loss(
      self,
      resulting_dict: Dict[Text, tf.Tensor],
      pred_semantic_logits: tf.Tensor,
      ground_truth_semantic: tf.Tensor,
      ground_truth_semantic_weights: tf.Tensor,
      inverse_pixel_gt_mask_area: tf.Tensor,
      pixel_gt_non_void_mask: tf.Tensor):
    """Applies the semantic loss with optionally pixel subsampling.

    Args:
      resulting_dict: The resulting loss as a dict of tf.Tensor.
      pred_semantic_logits: A [batch, output_height, output_width, channel]
        float32, the predicted semantic logits.
      ground_truth_semantic: A [batch, input_height, input_width] int32
        tensor, indicating the semantic class each pixel belongs to.
      ground_truth_semantic_weights: A [batch, input_height, input_width]
        float32 tensor, indicating the loss weighting of each pixel.
      inverse_pixel_gt_mask_area: A [batch, output_height * output_width]
        float32 tensor, containing the inverse of ground truth mask area for
        each pixel. More specifically, the value equals
        output_height * output_width / pixel_gt_mask_area. The mask area is
        normalized by the total area (output_height * output_width) at the
        current output resolution, so that the mask areas are roughly invariant
        to the setting of different output strides.
      pixel_gt_non_void_mask: A [batch, output_height * output_width] float32
        tensor, with value 0.0 and 1.0 only, indicating if a pixel does not
        belong to the void class.
    """
    if self._semantic_loss_weight == 0:
      return
    if self._semantic_sample_k == 0:
      # Perform the default semantic loss with all pixels.
      resulting_dict[common.SEMANTIC_LOSS] = (
          self._semantic_loss(
              {
                  common.GT_SEMANTIC_KEY:
                      ground_truth_semantic,
                  common.SEMANTIC_LOSS_WEIGHT_KEY:
                      ground_truth_semantic_weights
              }, {
                  common.PRED_SEMANTIC_LOGITS_KEY:
                      pred_semantic_logits
              }) * self._semantic_loss_weight)
      return
    # Sample prediction and ground truth.
    # Firstly, as prediction may be in a lower resolution, we perform
    # downsampling if needed.
    gt_shape = ground_truth_semantic.get_shape().as_list()
    pred_shape = pred_semantic_logits.get_shape().as_list()
    if gt_shape[1:3] != pred_shape[1:3]:
      ground_truth_semantic = utils.strided_downsample(
          ground_truth_semantic, pred_shape[1:3])
      ground_truth_semantic_weights = utils.strided_downsample(
          ground_truth_semantic_weights, pred_shape[1:3])

    # Flatten all tensors with 1D spatial dimension.
    pred_semantic_logits_flat = tf.reshape(
        pred_semantic_logits,
        [-1, pred_shape[1] * pred_shape[2], pred_shape[3]])
    ground_truth_semantic_flat = tf.reshape(ground_truth_semantic,
                                            [-1, pred_shape[1] * pred_shape[2]])
    ground_truth_semantic_weights_flat = tf.reshape(
        ground_truth_semantic_weights, [-1, pred_shape[1] * pred_shape[2]])

    # Compute pixel space sampling indices according to mask area.
    sample_logits = (
        tf.math.log(inverse_pixel_gt_mask_area) *
        self._semantic_sample_temperature)
    sample_logits += (1 - pixel_gt_non_void_mask) * _SOFTMAX_MASKING_CONSTANT
    sample_indices = _gumbel_topk_sample(sample_logits,
                                         self._semantic_sample_k)
    subsampled_pred_semantic_logits_flat = tf.gather(
        pred_semantic_logits_flat, sample_indices, batch_dims=1)
    subsampled_ground_truth_semantic_flat = tf.gather(
        ground_truth_semantic_flat, sample_indices, batch_dims=1)
    subsampled_ground_truth_semantic_weights_flat = tf.gather(
        ground_truth_semantic_weights_flat, sample_indices, batch_dims=1)
    # Auxiliary semantic_loss.
    resulting_dict[common.SEMANTIC_LOSS] = (
        self._semantic_loss(
            {
                common.GT_SEMANTIC_KEY:
                    subsampled_ground_truth_semantic_flat,
                common.SEMANTIC_LOSS_WEIGHT_KEY:
                    subsampled_ground_truth_semantic_weights_flat
            }, {
                common.PRED_SEMANTIC_LOGITS_KEY:
                    subsampled_pred_semantic_logits_flat
            }) * self._semantic_loss_weight)

  def call(
      self, inputs: Tuple[Dict[Text, tf.Tensor], Dict[Text, tf.Tensor]]
  ) -> Dict[Text, tf.Tensor]:
    """Computes the MaX-DeepLab losses.

    Args:
      inputs: A tuple of two dicts (y_true, y_pred):
      - y_true: A dict of tensors providing ground-truth information, containing
         - common.GT_SEMANTIC_KEY: A [batch, height, width] int32 tf.Tensor, the
           semantic label map.
         - common.GT_THING_ID_MASK_KEY: A [batch, height, width] int32
           tf.Tensor. It assigns each non-crowd thing instance a unique mask-ID
           label, starting from 0. Unassigned pixels are set to -1.
         - common.GT_THING_ID_CLASS_KEY: A [batch, max_thing_id] int32
           tf.Tensor. It contains semantic ID of each instance assigned to
           thing_id_mask. The remaining (max_thing_id - num_things) elements are
           set to -1.
      - y_pred: A dict of tensors providing predictions.
         - common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY: A [batch_size,
           output_height, output_width, channels] float32 tensor.
         - common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY: A [batch_size,
           output_height, output_width, num_mask_slots] float32 tensor, the
           logits that a pixel belongs to a mask slot.
         - common.PRED_TRANSFORMER_CLASS_LOGITS_KEY: A [batch_size,
           num_mask_slots, num_thing_stuff_classes + 1] float32 tensor, the
           logits that a mask belongs to a semantic class (including thing,
           stuff, and void)

    Returns:
      The loss as a dict of tf.Tensor, optionally containing the following:
      - common.PQ_STYLE_LOSS_CLASS_TERM: [batch].
      - common.PQ_STYLE_LOSS_MASK_DICE_TERM: [batch].
      - common.MASK_ID_CROSS_ENTROPY_LOSS: [batch].
      - common.INSTANCE_DISCRIMINATION_LOSS: [batch].
      - common.SEMANTIC_LOSS: [batch].

    Raises:
      ValueError: A ValueError is raised if auxiliary_output_number > 0 but
        common.PRED_AUXILIARY_OUTPUTS does not exist in prediction.
      ValueError: A ValueError is raised if the length of auxiliary_outputs
        does not match auxiliary_output_number.
    """
    y_true, y_pred = inputs

    # total_output_number includes main output and auxiliary outputs.
    total_output_number = self._auxiliary_output_number + 1
    # Define result dicts for main output and auxiliary outputs.
    resulting_dict_list = [
        {} for auxiliary_idx in range(total_output_number)
    ]

    pixel_feature = y_pred[common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY]
    _, output_height, output_width, _ = (
        pixel_feature.get_shape().as_list())
    batch_size = utils.resolve_batch_size(pixel_feature)
    pixel_feature = tf.reshape(
        pixel_feature, [batch_size, output_height * output_width, -1])

    # Pre-process the ground truth.
    (pixel_gt_thing_mask, pixel_gt_non_void_mask, pixel_gt_mask_id_one_hot,
     mask_gt_semantic_map, mask_gt_non_void_mask, mask_gt_semantic_one_hot,
     mask_gt_area, inverse_pixel_gt_mask_area) = self._pre_process_ground_truth(
         y_true, output_height, output_width)
    pixel_gt_non_void_mask_expanded = tf.expand_dims(
        pixel_gt_non_void_mask, axis=-1)

    if self._auxiliary_output_number > 0:
      if common.PRED_AUXILIARY_OUTPUTS not in y_pred:
        raise ValueError('auxiliary_output_number > 0 but'
                         ' common.PRED_AUXILIARY_OUTPUTS does not exist in'
                         ' y_pred!')

      auxiliary_outputs = y_pred[common.PRED_AUXILIARY_OUTPUTS]
      if len(auxiliary_outputs) != self._auxiliary_output_number:
        raise ValueError('The length of auxiliary_outputs must be equal to'
                         ' self._auxiliary_output_number!')

    # Extract pixel_space_mask_logits.
    pixel_space_mask_logits = y_pred[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY]
    pixel_space_mask_logits = tf.reshape(
        pixel_space_mask_logits,
        [batch_size, output_height * output_width, self._num_mask_slots])

    # Extract transformer_class_logits.
    transformer_class_logits = y_pred[common.PRED_TRANSFORMER_CLASS_LOGITS_KEY]

    # Put all outputs into a list.
    pixel_feature_list = [pixel_feature]
    pixel_space_mask_logits_list = [pixel_space_mask_logits]
    transformer_class_logits_list = [transformer_class_logits]

    for auxiliary_idx in range(self._auxiliary_output_number):
      auxiliary_pixel_feature = auxiliary_outputs[auxiliary_idx][
          common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY]
      auxiliary_pixel_feature = tf.reshape(
          auxiliary_pixel_feature,
          [batch_size, output_height * output_width, -1])
      pixel_feature_list.append(auxiliary_pixel_feature)

      auxiliary_pixel_space_mask_logits = auxiliary_outputs[auxiliary_idx][
          common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY]
      auxiliary_pixel_space_mask_logits = tf.reshape(
          auxiliary_pixel_space_mask_logits,
          [batch_size, output_height * output_width, self._num_mask_slots])
      pixel_space_mask_logits_list.append(auxiliary_pixel_space_mask_logits)

      auxiliary_transformer_class_logits = auxiliary_outputs[auxiliary_idx][
          common.PRED_TRANSFORMER_CLASS_LOGITS_KEY]
      transformer_class_logits_list.append(auxiliary_transformer_class_logits)

    for auxiliary_idx in range(total_output_number):
      if self._instance_discrimination_sample_k > 0:
        self._call_pixel_contrastive_instance_discrimination_loss(
            resulting_dict=resulting_dict_list[auxiliary_idx],
            pixel_feature=pixel_feature_list[auxiliary_idx],
            pixel_gt_mask_id_one_hot=pixel_gt_mask_id_one_hot,
            inverse_pixel_gt_mask_area=inverse_pixel_gt_mask_area,
            pixel_gt_non_void_mask=pixel_gt_non_void_mask)
      else:
        self._call_original_instance_discrimination_loss(
            resulting_dict=resulting_dict_list[auxiliary_idx],
            pixel_feature=pixel_feature_list[auxiliary_idx],
            pixel_gt_mask_id_one_hot=pixel_gt_mask_id_one_hot,
            mask_gt_area=mask_gt_area,
            mask_gt_non_void_mask=mask_gt_non_void_mask,
            pixel_gt_thing_mask=pixel_gt_thing_mask)

    pixel_space_mask_probs = tf.nn.softmax(pixel_space_mask_logits, axis=-1)

    # Compute the mask similarity between all ground truth masks and all
    # predicted masks.
    mask_similarity = _mask_similarity(
        pixel_gt_mask_id_one_hot,
        pixel_space_mask_probs * pixel_gt_non_void_mask_expanded,
        metric='dice')

    # Compute the class similarity by multiplying the ground truth one hot
    # encoding with the predicted probability distribution. This is done between
    # all ground truth masks and all predicted masks.
    transformer_class_probs = tf.nn.softmax(
        transformer_class_logits, axis=-1)[:, :, :-1]
    class_similarity = tf.einsum(
        'bij,bkj->bik', mask_gt_semantic_one_hot, transformer_class_probs)

    # Compute hungarian matching weights. We take the negative here since the
    # hungarian matching algorithm looks for the matching with the least total
    # weight.
    hungarian_weights = - mask_similarity * class_similarity
    mask_gt_non_void_mask_expanded_2 = tf.expand_dims(
        mask_gt_non_void_mask, axis=2)

    # Mask the void ground truth masks (in the rows) so that they do not affect
    # the result of the hungarian matching.
    if self._num_mask_slots >= self._pixel_gt_num_mask_id:
      # If the number of mask slots (number of columns) is larger than the
      # constant number of ground truth masks (number of rows), the
      # nonsquare_hungarian_matching will pad the rows with
      # _MATCHING_NEGATIVE_CONSTANT. In this case, we can fill in the void mask
      # rows with _MATCHING_NEGATIVE_CONSTANT too, then the void mask rows will
      # be ignored too, according to the hungarian matching property.
      hungarian_weights = (
          hungarian_weights * mask_gt_non_void_mask_expanded_2 +
          (1 - mask_gt_non_void_mask_expanded_2) * _MATCHING_NEGATIVE_CONSTANT)
    else:
      # If the number of mask slots (number of columns) is smaller than the
      # constant number of ground truth masks (number of rows), the
      # nonsquare_hungarian_matching will pad the columns with
      # _MATCHING_NEGATIVE_CONSTANT. In this case, we should fill in the void
      # mask rows with _MATCHING_POSITIVE_CONSTANT here, then the void mask rows
      # will have a huge cost compared with existing non-void mask rows, so that
      # the predicted masks will prefer matching with existing non-void masks
      # rather than the padded void masks, according to the hungarian matching
      # property.
      hungarian_weights = (
          hungarian_weights * mask_gt_non_void_mask_expanded_2 +
          (1 - mask_gt_non_void_mask_expanded_2) * _MATCHING_POSITIVE_CONSTANT)

    # Perform the hungarian matching algorithm.
    full_permutation, nonsquare_permutation = (
        nonsquare_hungarian_matching(hungarian_weights))

    # Extract the permutation (matching) between all existing non-void ground
    # truth masks and the matched predicted masks.
    matched_permutation = (
        nonsquare_permutation * mask_gt_non_void_mask_expanded_2)
    # The matched mask dice scores for each mask slot. The scores will be used
    # as a loss weight for the PQ-style loss class term after the stop_gradient.
    matched_mask_dice = tf.reduce_max(
        mask_similarity * matched_permutation, axis=-2)
    matched_mask_dice = tf.stop_gradient(matched_mask_dice)

    # The matched class probabilities for each ground truth mask. The
    # probabilities will be used as a loss weight for the PQ-style loss mask
    # dice term after the stop_gradient.
    matched_class_prob = tf.reduce_max(
        class_similarity * matched_permutation, axis=-1)
    matched_class_prob = tf.stop_gradient(matched_class_prob)

    # Extract the index of the matched mask slot for each ground truth mask.
    matched_mask_slot_indices = tf.math.argmax(
        nonsquare_permutation, axis=-1, output_type=tf.dtypes.int32)

    full_num_mask_slots = full_permutation.get_shape().as_list()[-1]

    permuted_full_pixel_space_mask_logits_list = []
    for auxiliary_idx in range(total_output_number):
      # Pad the pixel_space_mask_logits so that it is compatible with the
      # permutation matrix.
      full_pixel_space_mask_logits = tf.pad(
          pixel_space_mask_logits_list[auxiliary_idx],
          [[0, 0], [0, 0], [0, full_num_mask_slots - self._num_mask_slots]],
          constant_values=_SOFTMAX_MASKING_CONSTANT)
      # Permute the pixel space mask logits with the permutation matrix, which
      # converts the mask slot indices to the ground truth indices.
      permuted_full_pixel_space_mask_logits = tf.einsum(
          'bpi,bji->bpj', full_pixel_space_mask_logits, full_permutation)
      permuted_full_pixel_space_mask_logits_list.append(
          permuted_full_pixel_space_mask_logits)

    # Pad the class probabilities too.
    full_matched_class_prob = tf.pad(
        matched_class_prob,
        [[0, 0], [0, full_num_mask_slots - self._pixel_gt_num_mask_id]])
    # We only compute dice loss term on non-void ground truth masks.
    mask_dice_term_loss_weight = tf.pad(
        mask_gt_non_void_mask,
        [[0, 0], [0, full_num_mask_slots - self._pixel_gt_num_mask_id]])
    # Use the class probabilities as the loss weight for the mask dice term. In
    # addition, we set a lower bound, 1e-5, for the mask dice term loss weight.
    # Otherwise, if a loss weight is accidentally zero, the dice loss will treat
    # it as void and use an incorrect denominator or normalizing constant for
    # the loss.
    mask_dice_term_loss_weight *= tf.maximum(full_matched_class_prob, 1e-5)

    # Pad the one hot encoding too.
    full_pixel_gt_mask_id_one_hot = tf.pad(
        pixel_gt_mask_id_one_hot,
        [[0, 0], [0, 0], [0, full_num_mask_slots - self._pixel_gt_num_mask_id]])

    for auxiliary_idx in range(total_output_number):
      if self._pq_style_loss_weight > 0.0:
        # Mask_dice_term_modifier balances the mask_dice_term and the class_term
        # of the PQ-style loss to have the same weight and normalizating
        # constant.
        resulting_dict_list[auxiliary_idx][
            common.PQ_STYLE_LOSS_MASK_DICE_TERM] = (
                self._pq_style_loss_mask_dice_term(
                    {_GT_KEY: full_pixel_gt_mask_id_one_hot}, {
                        _PRED_KEY:
                            permuted_full_pixel_space_mask_logits_list[
                                auxiliary_idx],
                        _WEIGHT_KEY:
                            mask_dice_term_loss_weight
                    }) *
                (self._pq_style_loss_weight * self._mask_dice_term_modifier))

      # Mask-ID cross entropy loss shares the same ground truth and logits as
      # the dice loss term, but with different weights.
      if self._mask_id_cross_entropy_loss_weight > 0.0:
        resulting_dict_list[auxiliary_idx][
            common.MASK_ID_CROSS_ENTROPY_LOSS] = (
                self._mask_id_cross_entropy_loss(
                    {_GT_KEY: full_pixel_gt_mask_id_one_hot}, {
                        _PRED_KEY:
                            permuted_full_pixel_space_mask_logits_list[
                                auxiliary_idx],
                        _WEIGHT_KEY:
                            pixel_gt_non_void_mask
                    }) * self._mask_id_cross_entropy_loss_weight)

    # Generate a pseudo ground truth for transformer_class_logits.
    mask_slot_semantic_one_hot = _generate_mask_slot_semantic_one_hot(
        matched_mask_slot_indices, mask_gt_semantic_map,
        self._num_mask_slots, self._thing_stuff_class_ids)

    # Compute the positive mask and the negative mask.
    mask_slot_positive_mask = tf.cast(tf.equal(tf.reduce_max(
        mask_slot_semantic_one_hot, axis=-1), 1.0), tf.float32)
    mask_slot_negative_mask = 1.0 - mask_slot_positive_mask

    # Compute the overlap ratio between all predicted masks and the void region.
    # This void ratio will be used as a weight for the negative class term.
    mask_void_ratio = tf.stop_gradient(_mask_similarity(
        1.0 - pixel_gt_non_void_mask_expanded,
        pixel_space_mask_probs,
        'intersection_over_prediction'))
    mask_void_ratio = tf.squeeze(mask_void_ratio, axis=1)

    # Use the matched mask dice scores as the weights for the positive class
    # terms. For the negative class terms, we reduce the penalty for a mask slot
    # class term if the mask prediction overlaps a lot with void regions.
    transformer_class_loss_weight = (
        mask_slot_positive_mask * tf.maximum(matched_mask_dice, 1e-5) +
        mask_slot_negative_mask * tf.maximum(mask_void_ratio, 1e-5))

    # Concatenate the void mask in the last channel, constructing the final
    # ground truth one hot label with (thing + stuff + void) channels.
    transformer_class_one_hot = tf.concat(
        [mask_slot_semantic_one_hot,
         tf.expand_dims(mask_slot_negative_mask, axis=-1)], axis=-1)

    for auxiliary_idx in range(total_output_number):
      # Apply the PQ-style loss class term.
      if self._pq_style_loss_weight > 0.0:
        resulting_dict_list[auxiliary_idx][common.PQ_STYLE_LOSS_CLASS_TERM] = (
            self._pq_style_loss_class_term(
                {_GT_KEY: transformer_class_one_hot},
                {_PRED_KEY: transformer_class_logits_list[auxiliary_idx],
                 _WEIGHT_KEY: transformer_class_loss_weight}) *
            self._pq_style_loss_weight)

    # Aggregate resulting_dict_list into one. We assume the first element is the
    # main output.
    resulting_dict = resulting_dict_list[0]
    for auxiliary_idx in range(self._auxiliary_output_number):
      auxiliary_resulting_dict = resulting_dict_list[auxiliary_idx + 1]
      for loss_key in [common.PQ_STYLE_LOSS_CLASS_TERM,
                       common.PQ_STYLE_LOSS_MASK_DICE_TERM,
                       common.MASK_ID_CROSS_ENTROPY_LOSS,
                       common.INSTANCE_DISCRIMINATION_LOSS]:
        if loss_key in auxiliary_resulting_dict:
          resulting_dict['aux_' + str(auxiliary_idx) + loss_key] = (
              auxiliary_resulting_dict[loss_key])

    self._call_semantic_loss(
        resulting_dict=resulting_dict,
        pred_semantic_logits=y_pred[common.PRED_SEMANTIC_LOGITS_KEY],
        ground_truth_semantic=y_true[common.GT_SEMANTIC_KEY],
        ground_truth_semantic_weights=y_true[common.SEMANTIC_LOSS_WEIGHT_KEY],
        inverse_pixel_gt_mask_area=inverse_pixel_gt_mask_area,
        pixel_gt_non_void_mask=pixel_gt_non_void_mask)

    return resulting_dict
