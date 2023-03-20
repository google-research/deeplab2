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

"""This file contains functions to preprocess images and labels."""

import tensorflow as tf

from deeplab2.data.preprocessing import autoaugment_utils
from deeplab2.data.preprocessing import preprocess_utils

# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5

_MEAN_PIXEL = [127.5, 127.5, 127.5]


def _pad_image_and_label(image,
                         label,
                         offset_height,
                         offset_width,
                         target_height,
                         target_width,
                         ignore_label=None):
  """Pads the image and the label to the given size.

  Args:
    image: A tf.Tensor of shape [height, width, channels].
    label: A tf.Tensor of shape [height, width, 1] or None.
    offset_height: The number of rows of zeros to add on top of the image and
      label.
    offset_width: The number of columns of zeros to add on the left of the image
      and label.
    target_height: The total height after padding.
    target_width: The total width after padding.
    ignore_label: The ignore_label for the label. Must only be set when label is
      given.

  Returns:
    The padded image and label as a tuple (padded_image, padded_label).

  Raises:
    tf.errors.InvalidArgumentError: An error occurs if the padding configuration
      is invalid.
    ValueError: An error occurs if label is given without an ignore_label.
  """
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  original_dtype = image.dtype
  if original_dtype not in (tf.float32, tf.float64):
    image = tf.cast(image, tf.float32)

  bottom_padding = target_height - offset_height - height
  right_padding = target_width - offset_width - width

  assert_bottom_padding = tf.assert_greater(
      bottom_padding, -1,
      'The padding configuration is not valid. Please either increase the '
      'target size or reduce the padding offset.')
  assert_right_padding = tf.assert_greater(
      right_padding, -1, 'The padding configuration is not valid. Please either'
      ' increase the target size or reduce the padding offset.')
  with tf.control_dependencies([assert_bottom_padding, assert_right_padding]):
    paddings = [[offset_height, bottom_padding], [offset_width, right_padding],
                [0, 0]]

    image = image - _MEAN_PIXEL
    image = tf.pad(image, paddings)
    image = image + _MEAN_PIXEL
    image = tf.cast(image, original_dtype)

    if label is not None:
      if ignore_label is None:
        raise ValueError(
            'If a label is given, the ignore label must be set too.')
      label = tf.pad(label, paddings, constant_values=ignore_label)

    return image, label


def _update_max_resize_value(max_resize_value, crop_size, is_inference=False):
  """Checks and may update max_resize_value.

  Args:
    max_resize_value: A 2-tuple of (height, width), maximum allowed value after
      resize. If a single element is given, then height and width share the same
      value. None, empty or having 0 indicates no maximum value will be used.
    crop_size: A 2-tuple of (height, width), crop size used.
    is_inference: Boolean, whether the model is performing inference or not.

  Returns:
    Updated max_resize_value.
  """
  max_resize_value = preprocess_utils.process_resize_value(max_resize_value)
  if max_resize_value is None and is_inference:
    # During inference, default max_resize_value to crop size to allow
    # model taking input images with larger sizes.
    max_resize_value = crop_size

  if max_resize_value is None:
    return None

  if max_resize_value[0] > crop_size[0] or max_resize_value[1] > crop_size[1]:
    raise ValueError(
        'Maximum resize value provided (%s) exceeds model crop size (%s)' %
        (max_resize_value, crop_size))
  return max_resize_value


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               prev_image=None,
                               prev_label=None,
                               depth=None,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=None,
                               ignore_depth=None,
                               is_training=True,
                               autoaugment_policy_name=None):
  """Preprocesses the image and label.

  Args:
    image: A tf.Tensor containing the image with shape [height, width, 3].
    label: A tf.Tensor containing the label with shape [height, width, 1] or
      None.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    prev_image: An optional tensor of shape [image_height, image_width, 3].
    prev_label: An optional tensor of shape [label_height, label_width, 1].
    depth: An optional tensor of shape [label_height, label_width, 1].
    min_resize_value: A 2-tuple of (height, width), desired minimum value after
      resize. If a single element is given, then height and width share the same
      value. None, empty or having 0 indicates no minimum value will be used.
    max_resize_value: A 2-tuple of (height, width), maximum allowed value after
      resize. If a single element is given, then height and width share the same
      value. None, empty or having 0 indicates no maximum value will be used.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor for random scale augmentation.
    max_scale_factor: Maximum scale factor for random scale augmentation.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    ignore_depth: The depth value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    autoaugment_policy_name: String, autoaugment policy name. See
      autoaugment_policy.py for available policies.

  Returns:
    resized_image: The resized input image without other augmentations as a
      tf.Tensor.
    processed_image: The preprocessed image as a tf.Tensor.
    label: The preprocessed groundtruth segmentation label as a tf.Tensor.
    preprocessed_prev_image: The preprocessed prev_image as a tf.Tensor.
    prev_label: The preprocessed prev_label as a tf.Tensor.
    depth: The preprocessed depth as a tf.Tensor.

  Raises:
    ValueError: Ground truth label not provided during training.
    ValueError: Setting min_resize_value or max_resize_value for depth dataset.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')

  image.get_shape().assert_is_compatible_with(tf.TensorShape([None, None, 3]))

  # Keep reference to original image.
  resized_image = image
  if prev_image is not None:
    image = tf.concat([image, prev_image], axis=2)
  processed_image = tf.cast(image, tf.float32)
  processed_prev_image = None

  if label is not None:
    label.get_shape().assert_is_compatible_with(tf.TensorShape([None, None, 1]))
    if prev_label is not None:
      label = tf.concat([label, prev_label], axis=2)
    label = tf.cast(label, tf.int32)

  if depth is not None:
    if (any(value != 0 for value in min_resize_value) or
        any(value != 0 for value in max_resize_value)):
      raise ValueError(
          'Depth prediction with non-zero min_resize_value or max_resize_value'
          'is not supported.')
    depth.get_shape().assert_is_compatible_with(tf.TensorShape([None, None, 1]))
    depth = tf.cast(depth, tf.int32)

  # Resize image and label to the desired range.
  if any([min_resize_value, max_resize_value, not is_training]):
    max_resize_value = _update_max_resize_value(
        max_resize_value,
        crop_size=(crop_height, crop_width),
        is_inference=not is_training)

    processed_image, label = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    if prev_image is None:
      resized_image = tf.identity(processed_image)
    else:
      resized_image, _ = tf.split(processed_image, 2, axis=2)

  if prev_image is not None:
    processed_image, processed_prev_image = tf.split(processed_image, 2, axis=2)

  if prev_label is not None:
    label, prev_label = tf.split(label, 2, axis=2)

  if not is_training:
    image_height = tf.shape(processed_image)[0]
    image_width = tf.shape(processed_image)[1]

    offset_height = 0
    offset_width = 0
    image_before_padding = processed_image
    processed_image, label = _pad_image_and_label(processed_image, label,
                                                  offset_height, offset_width,
                                                  crop_height, crop_width,
                                                  ignore_label)
    processed_image.set_shape([crop_height, crop_width, 3])
    if label is not None:
      label.set_shape([crop_height, crop_width, 1])
    if prev_image is not None:
      processed_prev_image, prev_label = _pad_image_and_label(
          processed_prev_image, prev_label, offset_height, offset_width,
          crop_height, crop_width, ignore_label)
      processed_prev_image.set_shape([crop_height, crop_width, 3])
      if prev_label is not None:
        prev_label.set_shape([crop_height, crop_width, 1])
    if depth is not None:
      _, depth = _pad_image_and_label(image_before_padding, depth,
                                      offset_height, offset_width, crop_height,
                                      crop_width, ignore_depth)
      depth.set_shape([crop_height, crop_width, 1])
    return (resized_image, processed_image, label, processed_prev_image,
            prev_label, depth)

  # Data augmentation by randomly scaling the inputs.
  scale = preprocess_utils.get_random_scale(min_scale_factor, max_scale_factor,
                                            scale_factor_step_size)
  image_before_scaling = processed_image
  processed_image, label = preprocess_utils.randomly_scale_image_and_label(
      processed_image, label, scale)
  if processed_prev_image is not None:
    (processed_prev_image,
     prev_label) = preprocess_utils.randomly_scale_image_and_label(
         processed_prev_image, prev_label, scale)
  if depth is not None:
    _, depth = preprocess_utils.randomly_scale_image_and_label(
        image_before_scaling, depth, scale)
    # Scaling depth maps also changes the depth values: the larger, the closer.
    depth = tf.cast(depth, tf.float32)
    depth = depth / scale
    depth = tf.cast(depth, tf.int32)

  # Apply autoaugment if any.
  if autoaugment_policy_name:
    processed_image, label = _autoaugment_helper(processed_image, label,
                                                 ignore_label,
                                                 autoaugment_policy_name)
    if processed_prev_image is not None:
      processed_prev_image, prev_label = _autoaugment_helper(
          processed_prev_image, prev_label, ignore_label,
          autoaugment_policy_name)

  # Pad image and label to have dimensions >= [crop_height, crop_width].
  image_height = tf.shape(processed_image)[0]
  image_width = tf.shape(processed_image)[1]
  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Randomly crop the image and label.
  def _uniform_offset(margin):
    return tf.random.uniform([],
                             minval=0,
                             maxval=tf.maximum(margin, 1),
                             dtype=tf.int32)

  offset_height = _uniform_offset(crop_height - image_height)
  offset_width = _uniform_offset(crop_width - image_width)
  image_before_padding = processed_image
  processed_image, label = _pad_image_and_label(processed_image, label,
                                                offset_height, offset_width,
                                                target_height, target_width,
                                                ignore_label)
  if processed_prev_image is not None:
    processed_prev_image, prev_label = _pad_image_and_label(
        processed_prev_image, prev_label, offset_height, offset_width,
        target_height, target_width, ignore_label)

  if depth is not None:
    _, depth = _pad_image_and_label(image_before_padding, depth, offset_height,
                                    offset_width, target_height, target_width,
                                    ignore_depth)

  if processed_prev_image is not None:
    if depth is not None:
      (processed_image, label, processed_prev_image, prev_label,
       depth) = preprocess_utils.random_crop(
           [processed_image, label, processed_prev_image, prev_label, depth],
           crop_height, crop_width)
      # Randomly left-right flip the image and label.
      (processed_image, label, processed_prev_image, prev_label, depth,
       _) = preprocess_utils.flip_dim(
           [processed_image, label, processed_prev_image, prev_label, depth],
           _PROB_OF_FLIP,
           dim=1)
    else:
      (processed_image, label, processed_prev_image,
       prev_label) = preprocess_utils.random_crop(
           [processed_image, label, processed_prev_image, prev_label],
           crop_height, crop_width)
      # Randomly left-right flip the image and label.
      (processed_image, label, processed_prev_image, prev_label,
       _) = preprocess_utils.flip_dim(
           [processed_image, label, processed_prev_image, prev_label],
           _PROB_OF_FLIP,
           dim=1)
  else:
    processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width)
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

  return (resized_image, processed_image, label, processed_prev_image,
          prev_label, depth)


def _autoaugment_helper(image, label, ignore_label, policy_name):
  image = tf.cast(image, tf.uint8)
  label = tf.cast(label, tf.int32)
  image, label = autoaugment_utils.distort_image_with_autoaugment(
      image, label, ignore_label, policy_name)
  image = tf.cast(image, tf.float32)
  return image, label
