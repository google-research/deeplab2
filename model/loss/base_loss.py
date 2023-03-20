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

"""This file contains basic loss classes used in the DeepLab model."""

from typing import Text, Dict, Callable, Optional

import tensorflow as tf
from deeplab2.model import utils


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


def compute_mask_dice_loss(y_true: tf.Tensor,
                           y_pred: tf.Tensor,
                           prediction_activation='softmax') -> tf.Tensor:
  """Computes the Mask Dice loss between y_true and y_pred masks.

  Reference:
  [1] Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural
      networks for volumetric medical image segmentation. In: 3DV (2016)
      https://arxiv.org/abs/1606.04797

  Args:
    y_true: A tf.Tensor of shape [batch, height, width, channels] (or [batch,
      length, channels]) containing the ground-truth. The channel dimension
      indicates the mask ID in MaX-DeepLab, instead of a "class" dimension in
      the V-net paper. In our case, for all batch, height, width, (or batch,
      length) the [batch, height, width, :] (or [batch, length, :]) should be
      one-hot encodings only, with valid pixels having one and only one 1.0, and
      with void pixels being all 0.0. The valid pixels of the masks do not and
      should not overlap because of the non-overlapping definition of panoptic
      segmentation. The output loss is computed and normalized by valid (not
      void) pixels.
    y_pred: A tf.Tensor of shape [batch, height, width, channels] (or [batch,
      length, channels]) containing the prediction.
    prediction_activation: A String indicating activation function of the
      prediction. It should be either 'sigmoid' or 'softmax'.

  Returns:
    A tf.Tensor of shape [batch, channels] with the computed dice loss value.

  Raises:
      ValueError: An error occurs when prediction_activation is not either
        'sigmoid' or 'softmax'.
  """
  tf.debugging.assert_rank_in(
      y_pred, [3, 4], message='Input tensors y_pred must have rank 3 or 4.')
  tf.debugging.assert_rank_in(
      y_true, [3, 4], message='Input tensors y_true must have rank 3 or 4.')

  shape_list = y_true.shape.as_list()
  channels = shape_list[-1]
  batch = utils.resolve_batch_size(y_true)
  if prediction_activation == 'sigmoid':
    y_pred = tf.math.sigmoid(y_pred)
  elif prediction_activation == 'softmax':
    y_pred = tf.nn.softmax(y_pred, axis=-1)
  else:
    raise ValueError(
        "prediction_activation should be either 'sigmoid' or 'softmax'")

  y_true_flat = tf.reshape(y_true, [batch, -1, channels])
  # valid_flat indicates labeled pixels in the groudtruth. y_true is one-hot
  # encodings only, with valid pixels having one and only one 1.0, and with
  # invalid pixels having 0.0 values in all the channels. The valid pixels of
  # the masks do not overlap because of the non-overlapping definition of
  # panoptic segmentation.
  valid_flat = tf.reduce_sum(y_true_flat, axis=-1, keepdims=True)
  y_pred_flat = tf.reshape(
      y_pred, [batch, -1, channels]) * valid_flat
  # Use smooth = 1 to avoid division by zero when both y_pred and y_true are
  # zeros.
  smooth = 1.0
  intersection = 2 * tf.reduce_sum(y_pred_flat * y_true_flat, axis=1) + smooth
  denominator = (tf.reduce_sum(y_pred_flat, axis=1) +
                 tf.reduce_sum(y_true_flat, axis=1) + smooth)
  loss = 1. - tf.math.divide_no_nan(intersection, denominator)
  return loss


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


def encode_one_hot(gt: tf.Tensor,
                   num_classes: int,
                   weights: tf.Tensor,
                   ignore_label: Optional[int]):
  """Helper function for one-hot encoding of integer labels.

  Args:
    gt: A tf.Tensor providing ground-truth information. Integer type label.
    num_classes: An integer indicating the number of classes considered in the
      ground-truth. It is used as 'depth' in tf.one_hot().
    weights: A tf.Tensor containing weights information.
    ignore_label: An integer specifying the ignore label or None.

  Returns:
    gt: A tf.Tensor of one-hot encoded gt labels.
    weights: A tf.Tensor with ignore_label considered.
  """
  if ignore_label is not None:
    keep_mask = tf.cast(tf.not_equal(gt, ignore_label), dtype=tf.float32)
  else:
    keep_mask = tf.ones_like(gt, dtype=tf.float32)
  gt = tf.stop_gradient(tf.one_hot(gt, num_classes))
  weights = tf.multiply(weights, keep_mask)
  return gt, weights


def is_one_hot(gt: tf.Tensor, pred: tf.Tensor):
  """Helper function for checking if gt tensor is one-hot encoded or not.

  Args:
    gt: A tf.Tensor providing ground-truth information.
    pred: A tf.Tensor providing prediction information.

  Returns:
    A boolean indicating whether the gt is one-hot encoded (True) or
    in integer type (False).
  """
  gt_shape = gt.get_shape().as_list()
  pred_shape = pred.get_shape().as_list()
  # If the ground truth is one-hot encoded, the rank of the ground truth should
  # match that of the prediction. In addition, we check that the first
  # dimension, batch_size, and the last dimension, channels, should also match
  # the prediction. However, we still allow spatial dimensions, e.g., height and
  # width, to be different since we will downsample the ground truth if needed.
  return (len(gt_shape) == len(pred_shape) and
          gt_shape[0] == pred_shape[0] and gt_shape[-1] == pred_shape[-1])


def _ensure_topk_value_is_percentage(top_k_percentage: float):
  """Checks if top_k_percentage is between 0.0 and 1.0.

  Args:
    top_k_percentage: The floating point value to check.
  """
  if top_k_percentage < 0.0 or top_k_percentage > 1.0:
    raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                     '1.0, but %f was given' % top_k_percentage)


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
               num_classes: Optional[int],
               ignore_label: Optional[int],
               top_k_percent_pixels: float = 1.0,
               dynamic_weight: bool = False):
    """Initializes a top-k cross entropy loss.

    Args:
      gt_key: A key to extract the ground-truth tensor.
      pred_key: A key to extract the prediction tensor.
      weight_key: A key to extract the weight tensor.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or None.
      top_k_percent_pixels: An optional float specifying the percentage of
        pixels used to compute the loss. The value must lie within [0.0, 1.0].
      dynamic_weight: A boolean indicating whether the weights are determined
        dynamically w.r.t. the class confidence of each predicted mask.

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
    self._dynamic_weight = dynamic_weight

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the top-k cross-entropy loss.

    Args:
      y_true: A dict of tensors providing ground-truth information. The tensors
        can be either integer type or one-hot encoded. When is integer type, the
        shape can be either [batch, num_elements] or [batch, height, width].
        When one-hot encoded, the shape can be [batch, num_elements, channels]
        or [batch, height, width, channels].
      y_pred: A dict of tensors providing predictions. The tensors are of shape
        [batch, num_elements, channels] or [batch, height, width, channels]. If
        the prediction is 2D (with height and width), we allow the spatial
        dimension to be strided_height and strided_width. In this case, we
        downsample the ground truth accordingly.

    Returns:
      A tensor of shape [batch] containing the loss per image.

    Raises:
      ValueError: If the prediction is 1D (with the length dimension) but its
        length does not match that of the ground truth.
    """
    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    gt_shape = gt.get_shape().as_list()
    pred_shape = pred.get_shape().as_list()
    if self._dynamic_weight:
      weights = y_pred[self._weight_key]
    else:
      weights = y_true[self._weight_key]

    # Downsample the ground truth for 2D prediction cases.
    if len(pred_shape) == 4 and gt_shape[1:3] != pred_shape[1:3]:
      gt = utils.strided_downsample(gt, pred_shape[1:3])
      weights = utils.strided_downsample(weights, pred_shape[1:3])
    elif len(pred_shape) == 3 and gt_shape[1] != pred_shape[1]:
      # We don't support downsampling for 1D predictions.
      raise ValueError('The shape of gt does not match the shape of pred.')

    if is_one_hot(gt, pred):
      gt = tf.cast(gt, tf.float32)
    else:
      gt = tf.cast(gt, tf.int32)
      gt, weights = encode_one_hot(gt, self._num_classes, weights,
                                   self._ignore_label)
    pixel_losses = tf.keras.backend.categorical_crossentropy(
        gt, pred, from_logits=True)
    weighted_pixel_losses = tf.multiply(pixel_losses, weights)

    return compute_average_top_k_loss(weighted_pixel_losses,
                                      self._top_k_percent_pixels)


class FocalCrossEntropyLoss(tf.keras.losses.Loss):
  """This class contains code for focal cross-entropy."""

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               weight_key: Text,
               num_classes: Optional[int],
               ignore_label: Optional[int],
               focal_loss_alpha: float = 0.75,
               focal_loss_gamma: float = 0.0,
               background_channel_index: int = -1,
               dynamic_weight: bool = True):
    """Initializes a focal cross entropy loss.

    FocalCrossEntropyLoss supports focal-loss mode with integer
    or one-hot ground-truth labels.
    Reference:
    [1] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. Focal loss for
        dense object detection. In Proceedings of the IEEE International
        Conference on Computer Vision (ICCV). (2017)
        https://arxiv.org/abs/1708.02002

    Args:
      gt_key: A key to extract the ground-truth tensor.
      pred_key: A key to extract the prediction tensor.
      weight_key: A key to extract the weight tensor.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or None.
        Only effective when ground truth labels are in integer mode.
      focal_loss_alpha: An optional float specifying the coefficient that
        weights between positive (matched) and negative (unmatched) masks in
        focal loss. The positives are weighted by alpha, while the negatives
        are weighted by (1. - alpha). Default to 0.75.
      focal_loss_gamma: An optional float specifying the coefficient that
        weights probability (pt) term in focal loss. Focal loss = - ((1 - pt) ^
        gamma) * log(pt). Default to 0.0.
      background_channel_index: The index for background channel. When alpha
        is used, we assume the last channel is background and others are
        foreground. Default to -1.
      dynamic_weight: A boolean indicating whether the weights are determined
        dynamically w.r.t. the class confidence of each predicted mask.
    """
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super(FocalCrossEntropyLoss,
          self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    self._num_classes = num_classes
    self._ignore_label = ignore_label
    self._focal_loss_alpha = focal_loss_alpha
    self._focal_loss_gamma = focal_loss_gamma
    self._background_channel_index = background_channel_index
    self._gt_key = gt_key
    self._pred_key = pred_key
    self._weight_key = weight_key
    self._dynamic_weight = dynamic_weight

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the focal cross-entropy loss.

    Args:
      y_true: A dict of tensors providing ground-truth information. The tensors
        can be either integer type or one-hot encoded. When is integer type, the
        shape can be either [batch, num_elements] or [batch, height, width].
        When one-hot encoded, the shape can be [batch, num_elements, channels]
        or [batch, height, width, channels].
      y_pred: A dict of tensors providing predictions. The tensors are of shape
        [batch, num_elements, channels] or [batch, height, width, channels].

    Returns:
      A tensor of shape [batch] containing the loss per image.

    """

    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    if self._dynamic_weight:
      # Dynamic weights w.r.t. the class confidence of each predicted mask.
      weights = y_pred[self._weight_key]
    else:
      weights = y_true[self._weight_key]

    if is_one_hot(gt, pred):
      gt = tf.cast(gt, tf.float32)
    else:
      gt = tf.cast(gt, tf.int32)
      gt, weights = encode_one_hot(gt, self._num_classes, weights,
                                   self._ignore_label)
    pixel_losses = tf.nn.softmax_cross_entropy_with_logits(gt, pred)
    # Focal loss
    if self._focal_loss_gamma == 0.0:
      pixel_focal_losses = pixel_losses
    else:
      predictions = tf.nn.softmax(pred, axis=-1)
      pt = tf.reduce_sum(predictions * gt, axis=-1)
      pixel_focal_losses = tf.multiply(
          tf.pow(1.0 - pt, self._focal_loss_gamma), pixel_losses)

    if self._focal_loss_alpha >= 0:
      # alpha_weights = alpha * positive masks + (1 - alpha) * negative masks.
      alpha = self._focal_loss_alpha
      alpha_weights = (
          alpha * (1.0 - gt[..., self._background_channel_index])
          + (1 - alpha) * gt[..., self._background_channel_index])
      pixel_focal_losses = alpha_weights * pixel_focal_losses
    weighted_pixel_losses = tf.multiply(pixel_focal_losses, weights)
    weighted_pixel_losses = tf.reshape(
        weighted_pixel_losses, shape=(tf.shape(weighted_pixel_losses)[0], -1))
    # Compute mean loss over spatial dimension.
    num_non_zero = tf.reduce_sum(
        tf.cast(tf.not_equal(weighted_pixel_losses, 0.0), tf.float32), 1)
    loss_sum_per_sample = tf.reduce_sum(weighted_pixel_losses, 1)
    return tf.math.divide_no_nan(loss_sum_per_sample, num_non_zero)


class MaskDiceLoss(tf.keras.losses.Loss):
  """This class contains code to compute Mask Dice loss.

  The channel dimension in Mask Dice loss indicates the mask ID in MaX-DeepLab,
  instead of a "class" dimension in the original Dice loss.
  """

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               weight_key: Text,
               prediction_activation='softmax'):
    """Initializes a Mask Dice loss.

    Args:
      gt_key: A key to extract the ground-truth tensor.
      pred_key: A key to extract the pred tensor.
      weight_key: A key to extract the weight tensor.
      prediction_activation: A String indicating activation function of the
        prediction. It should be either 'sigmoid' or 'softmax'.
    """
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super(MaskDiceLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    self._gt_key = gt_key
    self._pred_key = pred_key
    self._weight_key = weight_key
    self._prediction_activation = prediction_activation

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the Mask Dice loss.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per sample.
    """
    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    # Dynamic weights w.r.t. the class confidence of each predicted mask.
    weights = y_pred[self._weight_key]
    weighted_dice_losses = tf.multiply(
        compute_mask_dice_loss(gt, pred, self._prediction_activation),
        weights)
    # Reduce_sum over the channels (i.e., number of masks).
    return tf.reduce_sum(weighted_dice_losses, axis=-1)


class SILogError(tf.keras.losses.Loss):
  """This class contains code to compute the SILog error.

  Scale invariant logarithmic (SILog) error was proposed for monocular depth
  estimation.

  Reference:
  Eigen, David, Christian Puhrsch, and Rob Fergus. "Depth map prediction from a
  single image using a multi-scale deep network." In NeurIPS, 2014.
  """

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               ignore_label: float):
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super().__init__(reduction=tf.keras.losses.Reduction.NONE)
    self._gt_key = gt_key
    self._pred_key = pred_key
    self._ignore_label = ignore_label

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the scale invariant logarithmic error.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per sample.
    """
    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    ignore_label = self._ignore_label

    def _compute_error(loss_input):
      gt, pred = loss_input
      label_mask = gt != ignore_label
      gt = tf.boolean_mask(gt, label_mask)
      pred = tf.boolean_mask(pred, label_mask)
      # Scale invariant logarithmic error.
      gt_log = tf.math.log(gt)
      pred_log = tf.math.log(pred)
      silog_error = (tf.reduce_mean(tf.square(gt_log - pred_log)) -
                     tf.square(tf.reduce_mean(gt_log - pred_log)))
      return silog_error

    return tf.map_fn(_compute_error, (gt, pred), fn_output_signature=tf.float32)


class RelativeSquaredError(tf.keras.losses.Loss):
  """This class contains code to compute the relative squared error.

  This class computes the relative squared error for monocular depth estimation.

  Reference:
  Uhrig, Jonas, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and
  Andreas Geiger. "Sparsity invariant cnns." In 3DV, 2017.
  """

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               ignore_label: float):
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super().__init__(reduction=tf.keras.losses.Reduction.NONE)
    self._gt_key = gt_key
    self._pred_key = pred_key
    self._ignore_label = ignore_label

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the relative squared error.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per sample.
    """
    gt = y_true[self._gt_key]
    pred = y_pred[self._pred_key]
    ignore_label = self._ignore_label

    def _compute_error(loss_input):
      gt, pred = loss_input
      label_mask = gt != ignore_label
      gt = tf.boolean_mask(gt, label_mask)
      pred = tf.boolean_mask(pred, label_mask)
      # Relative squared error.
      relative_squared_error = tf.sqrt(
          tf.reduce_mean(tf.square((gt - pred) / gt)))
      return relative_squared_error

    return tf.map_fn(_compute_error, (gt, pred), fn_output_signature=tf.float32)


class SILogPlusRelativeSquaredLoss(tf.keras.losses.Loss):
  """This class contains code to compute depth loss SILog + RelativeSquared.

  This depth loss function combines the scale invariant logarithmic (SILog)
  error and relative squared error, which was adopted in the ViP-DeepLab model.

  Reference:
  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." In CVPR 2021.
  """

  def __init__(self,
               gt_key: Text,
               pred_key: Text,
               ignore_label: float):
    # Implicit reduction might mess with tf.distribute.Strategy, hence we
    # explicitly reduce the loss.
    super().__init__(reduction=tf.keras.losses.Reduction.NONE)
    self._silog_error = SILogError(gt_key, pred_key, ignore_label)
    self._relativate_squared_error = RelativeSquaredError(
        gt_key, pred_key, ignore_label)

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Computes the loss for SILog + RelativeSquared.

    Args:
      y_true: A dict of tensors providing ground-truth information.
      y_pred: A dict of tensors providing predictions.

    Returns:
      A tensor of shape [batch] containing the loss per sample.
    """
    return self._silog_error(y_true, y_pred) + self._relativate_squared_error(
        y_true, y_pred)
