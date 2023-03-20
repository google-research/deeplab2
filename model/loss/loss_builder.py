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

"""This file contains loss builder classes used in the DeepLab model."""

import collections
from typing import Any, Dict, Text, Tuple, Optional

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.loss import base_loss
from deeplab2.model.loss import max_deeplab_loss


def _create_loss_and_weight(
    loss_options: config_pb2.LossOptions.SingleLossOptions, gt_key: Text,
    pred_key: Text, weight_key: Text, **kwargs: Any) -> tf.keras.losses.Loss:
  """Creates a loss and its weight from loss options.

  Args:
    loss_options: Loss options as defined by
      config_pb2.LossOptions.SingleLossOptions or None.
    gt_key: A key to extract the ground-truth from a dictionary.
    pred_key: A key to extract the prediction from a dictionary.
    weight_key: A key to extract the per-pixel weights from a dictionary.
    **kwargs: Additional parameters to initialize the loss.

  Returns:
    A tuple of an instance of tf.keras.losses.Loss and its corresponding weight
    as an integer.

  Raises:
    ValueError: An error occurs when the loss name is not a valid loss.
  """
  if loss_options is None:
    return None, 0
  if loss_options.name == 'softmax_cross_entropy':
    return base_loss.TopKCrossEntropyLoss(
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent,
        **kwargs), loss_options.weight  # pytype: disable=bad-return-type  # typed-keras
  elif loss_options.name == 'l1':
    return base_loss.TopKGeneralLoss(
        base_loss.mean_absolute_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight  # pytype: disable=bad-return-type  # typed-keras
  elif loss_options.name == 'mse':
    return base_loss.TopKGeneralLoss(
        base_loss.mean_squared_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight  # pytype: disable=bad-return-type  # typed-keras
  elif loss_options.name == 'depth_loss':
    return base_loss.SILogPlusRelativeSquaredLoss(
        gt_key,
        pred_key,
        **kwargs), loss_options.weight  # pytype: disable=bad-return-type  # typed-keras

  raise ValueError('Loss %s is not a valid loss.' % loss_options.name)


class DeepLabFamilyLoss(tf.keras.layers.Layer):
  """This class contains code to build and call losses for DeepLabFamilyLoss."""

  def __init__(
      self,
      loss_options: config_pb2.LossOptions,
      deeplab_options: config_pb2.ModelOptions,
      num_classes: Optional[int],
      ignore_label: Optional[int],
      ignore_depth: Optional[float],
      thing_class_ids: Tuple[int],
      auxiliary_output_number: int = 0):
    """Initializes the losses for the DeepLab family.

    Args:
      loss_options: Loss options as defined by config_pb2.LossOptions.
      deeplab_options: Model options as defined in config_pb2.ModelOptions.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or None.
      ignore_depth: An optional float specifying the ignore depth or None.
      thing_class_ids: A tuple of length [N] containing N thing indices.
      auxiliary_output_number: An integer specifying the number of auxiliary
        outputs. Only applicable to MaX-DeepLab.
    """
    super(DeepLabFamilyLoss, self).__init__(name='DeepLabFamilyLoss')

    # Single-term losses are losses that have only one loss term and thus each
    # loss function directly returns a single tensor as the loss value, as
    # opposed to multi-term losses that involve multiple terms and return a
    # dictionary of loss values.
    self._single_term_loss_func_and_weight_dict = collections.OrderedDict()
    self._extra_loss_names = [common.TOTAL_LOSS]

    if loss_options.HasField(common.CENTER_LOSS):
      self._single_term_loss_func_and_weight_dict[
          common.CENTER_LOSS] = _create_loss_and_weight(
              loss_options.center_loss, common.GT_INSTANCE_CENTER_KEY,
              common.PRED_CENTER_HEATMAP_KEY, common.CENTER_LOSS_WEIGHT_KEY)

    if loss_options.HasField(common.REGRESSION_LOSS):
      self._single_term_loss_func_and_weight_dict[
          common.REGRESSION_LOSS] = _create_loss_and_weight(
              loss_options.regression_loss, common.GT_INSTANCE_REGRESSION_KEY,
              common.PRED_OFFSET_MAP_KEY, common.REGRESSION_LOSS_WEIGHT_KEY)

    # Currently, only used for Motion-DeepLab.
    if loss_options.HasField(common.MOTION_LOSS):
      self._single_term_loss_func_and_weight_dict[
          common.MOTION_LOSS] = _create_loss_and_weight(
              loss_options.motion_loss, common.GT_FRAME_OFFSET_KEY,
              common.PRED_FRAME_OFFSET_MAP_KEY,
              common.FRAME_REGRESSION_LOSS_WEIGHT_KEY)

    # Next-frame regression loss used in ViP-DeepLab.
    if loss_options.HasField(common.NEXT_REGRESSION_LOSS):
      self._single_term_loss_func_and_weight_dict[
          common.NEXT_REGRESSION_LOSS] = _create_loss_and_weight(
              loss_options.next_regression_loss,
              common.GT_NEXT_INSTANCE_REGRESSION_KEY,
              common.PRED_NEXT_OFFSET_MAP_KEY,
              common.NEXT_REGRESSION_LOSS_WEIGHT_KEY)

    # Depth loss used in ViP-DeepLab.
    if loss_options.HasField(common.DEPTH_LOSS):
      self._single_term_loss_func_and_weight_dict[
          common.DEPTH_LOSS] = _create_loss_and_weight(
              loss_options.depth_loss,
              common.GT_DEPTH_KEY,
              common.PRED_DEPTH_KEY,
              common.DEPTH_LOSS_WEIGHT_KEY,
              ignore_label=ignore_depth)

    # Multi-term losses that return dictionaries of loss terms.
    self._multi_term_losses = []

    # MaXDeepLabLoss optionally returns five loss terms in total:
    # - common.PQ_STYLE_LOSS_CLASS_TERM
    # - common.PQ_STYLE_LOSS_MASK_DICE_TERM
    # - common.MASK_ID_CROSS_ENTROPY_LOSS
    # - common.INSTANCE_DISCRIMINATION_LOSS
    # - common.SEMANTIC_LOSS
    if any([loss_options.HasField('pq_style_loss'),
            loss_options.HasField('mask_id_cross_entropy_loss'),
            loss_options.HasField('instance_discrimination_loss')]):
      instance_discrimination_sample_k = (
          deeplab_options.max_deeplab.instance_discrimination_sample_k)
      instance_discrimination_sample_temperature = (
          deeplab_options.max_deeplab.instance_discrimination_sample_temperature
      )
      self._multi_term_losses.append(
          max_deeplab_loss.MaXDeepLabLoss(
              loss_options,
              num_classes,
              ignore_label,
              thing_class_ids,
              instance_discrimination_sample_k=instance_discrimination_sample_k,
              instance_discrimination_sample_temperature=(
                  instance_discrimination_sample_temperature),
              auxiliary_output_number=auxiliary_output_number))
    elif loss_options.HasField(common.SEMANTIC_LOSS):
      # The semantic loss will only be used when we do not use MaXDeepLabLoss,
      # which ensures the same behavior for other models (e.g.,
      # Panoptic-DeepLab).
      self._single_term_loss_func_and_weight_dict[
          common.SEMANTIC_LOSS] = _create_loss_and_weight(
              loss_options.semantic_loss,
              common.GT_SEMANTIC_KEY,
              common.PRED_SEMANTIC_LOGITS_KEY,
              common.SEMANTIC_LOSS_WEIGHT_KEY,
              num_classes=num_classes,
              ignore_label=ignore_label)

    for multi_term_loss in self._multi_term_losses:
      self._extra_loss_names += multi_term_loss.loss_terms

  def get_loss_names(self):
    # Keep track of all the keys that will be returned in self.call().
    loss_names = list(self._single_term_loss_func_and_weight_dict.keys())
    return loss_names + self._extra_loss_names

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the loss computations given ground-truth and predictions.

    The loss is computed for each sample separately. Currently, smoothed
    ground-truth labels are not supported.

    Args:
      y_true: A dictionary of tf.Tensor containing all ground-truth data to
        compute the loss. Depending on the configuration, the dict has to
        contain common.GT_SEMANTIC_KEY, and optionally
        common.GT_INSTANCE_CENTER_KEY, common.GT_INSTANCE_REGRESSION_KEY, and
        common.GT_FRAME_OFFSET_KEY.
      y_pred: A dicitionary of tf.Tensor containing all predictions to compute
        the loss. Depending on the configuration, the dict has to contain
        common.PRED_SEMANTIC_LOGITS_KEY, and optionally
        common.PRED_CENTER_HEATMAP_KEY, common.PRED_OFFSET_MAP_KEY, and
        common.PRED_FRAME_OFFSET_MAP_KEY.

    Returns:
      The loss as a dict of tf.Tensor, optionally containing the following:
      - common.SEMANTIC_LOSS: [batch].
      - common.CENTER_LOSS: [batch].
      - common.REGRESSION_LOSS: [batch].
      - common.MOTION_LOSS: [batch], the frame offset regression loss.
      - common.NEXT_REGRESSION_LOSS: [batch], the next regression loss.
      - common.PQ_STYLE_LOSS_CLASS_TERM: [batch].
      - common.PQ_STYLE_LOSS_MASK_DICE_TERM: [batch].
      - common.MASK_ID_CROSS_ENTROPY_LOSS: [batch].
      - common.INSTANCE_DISCRIMINATION_LOSS: [batch].

    Raises:
      AssertionError: If the keys of the resulting_dict do not match
        self.get_loss_names().
      AssertionError: The keys of the resulting_dict overlap with the keys of
        the loss_dict.
    """
    resulting_dict = collections.OrderedDict()

    # Single-term losses.
    for loss_name, func_and_weight in (
        self._single_term_loss_func_and_weight_dict.items()):
      loss_func, loss_weight = func_and_weight
      loss_value = loss_func(y_true, y_pred)
      resulting_dict[loss_name] = loss_value * loss_weight

    # Multi-term losses predict a dictionary, so we handle them differently.
    for multi_term_loss in self._multi_term_losses:
      loss_dict = multi_term_loss((y_true, y_pred))
      if not set(loss_dict).isdisjoint(resulting_dict):
        raise AssertionError('The keys of the resulting_dict overlap with the '
                             'keys of the loss_dict.')
      resulting_dict.update(loss_dict)

    # Also include the total loss in the resulting_dict.
    total_loss = tf.math.accumulate_n(list(resulting_dict.values()))
    resulting_dict[common.TOTAL_LOSS] = total_loss

    if sorted(resulting_dict.keys()) != sorted(self.get_loss_names()):
      raise AssertionError(
          'The keys of the resulting_dict should match self.get_loss_names().')
    return resulting_dict
