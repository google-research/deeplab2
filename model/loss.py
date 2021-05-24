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

"""This file contains loss classes used in the DeepLab model."""

from typing import Text, Dict, Callable, Any

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2


def compute_average_top_k_loss(loss: tf.Tensor,
                               top_k_percentage: float) -> tf.Tensor:
  """Computes the avaerage top-k loss per sample.

  Args:
    loss: A tf.Tensor with 2 or more dimensions of shape [batch, ...].
    top_k_percentage: A float representing the % of pixel that should be used
      for calculating the loss.

  Returns:
    A tensor of shape [batch] containing the mean top-k loss per sample. Due to
    the use of different tf.strategy, we return the loss per sample and require
    explicit averaging by the user.
  """
  loss = tf.reshape(loss, shape=(tf.shape(loss)[0], -1))

  if top_k_percentage != 1.0:
    num_elements_per_sample = tf.shape(loss)[1]
    top_k_pixels = tf.cast(
        tf.math.round(top_k_percentage *
                      tf.cast(num_elements_per_sample, tf.float32)), tf.int32)

    def top_k_1d(inputs):
      return tf.math.top_k(inputs, top_k_pixels, sorted=False)[0]
    loss = tf.map_fn(fn=top_k_1d, elems=loss)

  # Compute mean loss over spatial dimension.
  num_non_zero = tf.reduce_sum(tf.cast(tf.not_equal(loss, 0.0), tf.float32), 1)
  loss_sum_per_sample = tf.reduce_sum(loss, 1)
  return tf.math.divide_no_nan(loss_sum_per_sample, num_non_zero)


def mean_absolute_error(y_true: tf.Tensor,
                        y_pred: tf.Tensor,
                        force_keep_dims=False) -> tf.Tensor:
  """Computes the per-pixel mean absolute error for 3D and 4D tensors.

  Default reduction behavior: If a 3D tensor is used, no reduction is applied.
  In case of a 4D tensor, reduction is applied. This behavior can be overridden
  by force_keep_dims.
  Note: tf.keras.losses.mean_absolute_error always reduces the output by one
  dimension.

  Args:
    y_true: A tf.Tensor of shape [batch, height, width] or [batch, height,
      width, channels] containing the ground-truth.
    y_pred: A tf.Tensor of shape [batch, height, width] or [batch, height,
      width, channels] containing the prediction.
    force_keep_dims: A boolean flag specifying whether no reduction should be
      applied.

  Returns:
    A tf.Tensor with the mean absolute error.
  """
  tf.debugging.assert_rank_in(
      y_pred, [3, 4], message='Input tensors must have rank 3 or 4.')
  if len(y_pred.shape.as_list()) == 3 or force_keep_dims:
    return tf.abs(y_true - y_pred)
  else:
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=[3])


def mean_squared_error(y_true: tf.Tensor,
                       y_pred: tf.Tensor,
                       force_keep_dims=False) -> tf.Tensor:
  """Computes the per-pixel mean squared error for 3D and 4D tensors.

  Default reduction behavior: If a 3D tensor is used, no reduction is applied.
  In case of a 4D tensor, reduction is applied. This behavior can be overridden
  by force_keep_dims.
  Note: tf.keras.losses.mean_squared_error always reduces the output by one
  dimension.

  Args:
    y_true: A tf.Tensor of shape [batch, height, width] or [batch, height,
      width, channels] containing the ground-truth.
    y_pred: A tf.Tensor of shape [batch, height, width] or [batch, height,
      width, channels] containing the prediction.
    force_keep_dims: A boolean flag specifying whether no reduction should be
      applied.

  Returns:
    A tf.Tensor with the mean squared error.
  """
  tf.debugging.assert_rank_in(
      y_pred, [3, 4], message='Input tensors must have rank 3 or 4.')
  if len(y_pred.shape.as_list()) == 3 or force_keep_dims:
    return tf.square(y_true - y_pred)
  else:
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[3])


def _create_loss(loss_options: config_pb2.LossOptions.SingleLossOptions,
                 gt_key: Text, pred_key: Text, weight_key: Text,
                 **kwargs: Any) -> tf.keras.losses.Loss:
  """Creates a loss from loss options.

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
    return TopKCrossEntropyLoss(
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent,
        **kwargs), loss_options.weight
  elif loss_options.name == 'l1':
    return TopKGeneralLoss(
        mean_absolute_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight
  elif loss_options.name == 'mse':
    return TopKGeneralLoss(
        mean_squared_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight

  raise ValueError('Loss %s is not a valid loss.' % loss_options.name)


def _ensure_topk_value_is_percentage(top_k_percentage: float):
  """Checks if top_k_percentage is between 0.0 and 1.0.

  Args:
    top_k_percentage: The floating point value to check.
  """
  if top_k_percentage < 0.0 or top_k_percentage > 1.0:
    raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                     '1.0, but %f was given' % top_k_percentage)


class DeepLabFamilyLoss(tf.keras.losses.Loss):
  """This class contains code to build and call losses for DeepLabFamilyLoss."""

  def __init__(
      self,
      semantic_loss_options: config_pb2.LossOptions.SingleLossOptions,
      center_loss_options: config_pb2.LossOptions.SingleLossOptions,
      regression_loss_options: config_pb2.LossOptions.SingleLossOptions,
      motion_loss_options: config_pb2.LossOptions.SingleLossOptions,
      num_classes: int,
      ignore_label: int = 255):
    """Initializes the losses for Panoptic-DeepLab.

    Args:
      semantic_loss_options: Loss options as defined by
        config_pb2.LossOptions.SingleLossOptions.
      center_loss_options: Loss options as defined by
        config_pb2.LossOptions.SingleLossOptions or None.
      regression_loss_options: Loss options as defined by
        config_pb2.LossOptions.SingleLossOptions or None.
      motion_loss_options: Loss options as defined by
        config_pb2.LossOptions.SingleLossOptions or None.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or 'None'
        (default: 255).

    Raises:
      ValueError: An error occurs when the semantic loss is not defined
        correctly.
    """
    super(DeepLabFamilyLoss,
          self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    self._semantic_loss, self._semantic_weight = _create_loss(
        semantic_loss_options,
        common.GT_SEMANTIC_KEY,
        common.PRED_SEMANTIC_LOGITS_KEY,
        common.SEMANTIC_LOSS_WEIGHT_KEY,
        num_classes=num_classes,
        ignore_label=ignore_label)

    if self._semantic_loss is None:
      raise ValueError('The semantic loss must always be set.')

    self._center_loss, self._center_weight = _create_loss(
        center_loss_options,
        common.GT_INSTANCE_CENTER_KEY,
        common.PRED_CENTER_HEATMAP_KEY,
        common.CENTER_LOSS_WEIGHT_KEY)
    self._regression_loss, self._regression_weight = _create_loss(
        regression_loss_options,
        common.GT_INSTANCE_REGRESSION_KEY,
        common.PRED_OFFSET_MAP_KEY,
        common.REGRESSION_LOSS_WEIGHT_KEY)

    # Currently, only used for Motion-DeepLab.
    self._motion_loss, self._motion_weight = _create_loss(
        motion_loss_options, common.GT_FRAME_OFFSET_KEY,
        common.PRED_FRAME_OFFSET_MAP_KEY,
        common.FRAME_REGRESSION_LOSS_WEIGHT_KEY)

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
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
      The loss as tf.Tensor with shape [batch, 4] containing the following:
      - [:, 0] the semantic loss.
      - [:, 1] the center loss.
      - [:, 2] the offset regression loss.
      - [:, 3] the frame offset regression loss. Non-zero values when motion
      loss is used.
    """
    semantic_loss = self._semantic_loss(y_true, y_pred, self._semantic_weight)

    center_loss = tf.zeros_like(semantic_loss)
    if self._center_loss is not None:
      center_loss = self._center_loss(y_true, y_pred, self._center_weight)

    regression_loss = tf.zeros_like(semantic_loss)
    if self._regression_loss is not None:
      regression_loss = self._regression_loss(y_true, y_pred,
                                              self._regression_weight)

    motion_loss = tf.zeros_like(semantic_loss)
    if self._motion_loss is not None:
      motion_loss = self._motion_loss(y_true, y_pred, self._motion_weight)
    return tf.stack([semantic_loss, center_loss, regression_loss, motion_loss],
                    axis=1)


class TopKGeneralLoss(tf.keras.losses.Loss):
  """This class contains code to compute the top-k loss."""

  def __init__(self,
               loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
               gt_key: Text,
               pred_key: Text,
               weight_key: Text,
               top_k_percent_pixels: float = 1.0):
    """Initializes a top-k L1 loss.

    Args:
      loss_function: A callable loss function.
      gt_key: A key to extract the ground-truth tensor.
      pred_key: A key to extract the prediction tensor.
      weight_key: A key to extract the weight tensor.
      top_k_percent_pixels: An optional float specifying the percentage of
        pixels used to compute the loss. The value must lie within [0.0, 1.0].
    """
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super(TopKGeneralLoss,
          self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    _ensure_topk_value_is_percentage(top_k_percent_pixels)

    self._loss_function = loss_function
    self._top_k_percent_pixels = top_k_percent_pixels
    self._gt_key = gt_key
    self._pred_key = pred_key
    self._weight_key = weight_key

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the top-k loss.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per sample.
    """
    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    weights = y_true[self._weight_key]

    per_pixel_loss = self._loss_function(gt, pred)
    per_pixel_loss = tf.multiply(per_pixel_loss, weights)

    return compute_average_top_k_loss(per_pixel_loss,
                                      self._top_k_percent_pixels)


class TopKCrossEntropyLoss(tf.keras.losses.Loss):
  """This class contains code for top-k cross-entropy."""

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               weight_key: Text,
               num_classes: int,
               ignore_label: int = 255,
               top_k_percent_pixels: float = 1.0):
    """Initializes a top-k cross entropy loss.

    Args:
      gt_key: A key to extract the ground-truth tensor.
      pred_key: A key to extract the prediction tensor.
      weight_key: A key to extract the weight tensor.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or 'None'
        (default: 255).
      top_k_percent_pixels: An optional float specifying the percentage of
        pixels used to compute the loss. The value must lie within [0.0, 1.0].

    Raises:
      ValueError: An error occurs when top_k_percent_pixels is not between 0.0
        and 1.0.
    """
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super(TopKCrossEntropyLoss,
          self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    _ensure_topk_value_is_percentage(top_k_percent_pixels)

    self._num_classes = num_classes
    self._ignore_label = ignore_label
    self._top_k_percent_pixels = top_k_percent_pixels
    self._gt_key = gt_key
    self._pred_key = pred_key
    self._weight_key = weight_key

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the top-k cross-entropy loss.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per image.
    """
    gt = tf.cast(y_true[self._gt_key], tf.int32)
    pred = y_pred[self._pred_key]
    weights = y_true[self._weight_key]

    if self._ignore_label is not None:
      keep_mask = tf.cast(
          tf.not_equal(gt, self._ignore_label), dtype=tf.float32)
    else:
      keep_mask = tf.ones_like(gt, dtype=tf.float32)
    gt = tf.stop_gradient(tf.one_hot(gt, self._num_classes))

    pixel_losses = tf.keras.backend.categorical_crossentropy(
        gt, pred, from_logits=True)
    weights = tf.multiply(weights, keep_mask)
    weighted_pixel_losses = tf.multiply(pixel_losses, weights)

    return compute_average_top_k_loss(weighted_pixel_losses,
                                      self._top_k_percent_pixels)
