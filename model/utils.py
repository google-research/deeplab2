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

"""This file contains utility functions for the model code."""

from typing import Any, List, MutableMapping, MutableSequence, Optional

import tensorflow as tf

from deeplab2 import common

layers = tf.keras.layers

_PREDICTION_WITH_NEAREST_UPSAMPLING = [
    common.PRED_INSTANCE_KEY,
    common.PRED_INSTANCE_CENTER_KEY,
    common.PRED_INSTANCE_SCORES_KEY,
    common.PRED_PANOPTIC_KEY,
    common.PRED_SEMANTIC_KEY,
]

_PREDICTION_WITH_BILINEAR_UPSAMPLING = [
    common.PRED_SEMANTIC_PROBS_KEY,
]


def _scale_helper(value, scale):
  if isinstance(value, tf.Tensor):
    return tf.cast(
        (tf.cast(value, dtype=tf.float32) - 1.0) * scale + 1.0,
        dtype=tf.int32)
  else:
    return int((float(value) - 1.0) * scale + 1.0)


def scale_mutable_sequence(input_sequence: MutableSequence[int],
                           scale: float) -> MutableSequence[int]:
  return [_scale_helper(x, scale) for x in input_sequence]


def scale_int_list(int_list, scale):
  return [int(x * scale) for x in int_list]


def undo_image_preprocessing(image_in: tf.Tensor, method: str,
                             perform_crop: bool,
                             regions_to_crop: List[int],
                             output_shape: List[int]) -> tf.Tensor:
  """Undoes the image preprocessing.

  In particular, this function slices out the valid regions (determined by
  `regions_to_crop`) in the input when perform_crop is True. After
  that, we resize the results to the desired `output_shape`.

  Args:
    image_in: Input image Tensor with shape [batch, height, width, n_channels].
    method: Image resize method.
    perform_crop: Boolean, performing crop or not.
    regions_to_crop: The regions to crop [height, width]. Will only apply
      cropping at the bottom right.
    output_shape: Desired shape after resizing [height, width].

  Returns:
    Outputs after cropping (if perform_crop = True) and resizing.
  """
  if perform_crop:
    image_out = image_in[
        :, :regions_to_crop[0], :regions_to_crop[1], :]
  else:
    image_out = image_in
  return resize_align_corners(image_out, output_shape, method=method)


def undo_preprocessing_for_predictions(
    predictions: MutableMapping[str, Any], regions_to_crop: List[int],
    output_shape: List[int]) -> MutableMapping[str, Any]:
  """Undoes preprocessing for predictions.

  Args:
    predictions: A dictionary storing different types of predictions.
    regions_to_crop: The regions to crop [height, width]. Will only apply
      cropping at the bottom right.
    output_shape: Desired shape after resizing [height, width].

  Returns:
    predictions after cropping (if perform_crop = True) and resizing.
  """
  for pred_key in predictions.keys():
    if pred_key in _PREDICTION_WITH_NEAREST_UPSAMPLING:
      predictions[pred_key] = tf.squeeze(
          undo_image_preprocessing(
              tf.expand_dims(predictions[pred_key], 3),
              'nearest',
              perform_crop=True,
              regions_to_crop=regions_to_crop,
              output_shape=output_shape),
          axis=3)
    elif pred_key in _PREDICTION_WITH_BILINEAR_UPSAMPLING:
      predictions[pred_key] = undo_image_preprocessing(
          predictions[pred_key],
          'bilinear',
          perform_crop=True,
          regions_to_crop=regions_to_crop,
          output_shape=output_shape)
    else:
      # We only undo preprocessing for panoptic predictions. The
      # intermediate results (e.g., keypoint prediction) are not
      # processed. This is intentional, as not all models will generate
      # the intermediate results.
      continue
  return predictions


def add_zero_padding(input_tensor: tf.Tensor, kernel_size: int,
                     rank: int) -> tf.Tensor:
  """Adds zero-padding to the input_tensor."""
  pad_total = kernel_size - 1
  pad_begin = pad_total // 2
  pad_end = pad_total - pad_begin
  if rank == 3:
    return tf.pad(
        input_tensor,
        paddings=[[pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
  else:
    return tf.pad(
        input_tensor,
        paddings=[[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])


def resize_and_rescale_offsets(input_tensor: tf.Tensor, target_size):
  """Bilinearly resizes and rescales the offsets.

  Args:
    input_tensor: A tf.Tensor of shape [batch, height, width, 2].
    target_size: A list or tuple or 1D tf.Tensor that specifies the height and
      width after resizing.

  Returns:
    The input_tensor resized to shape `[batch, target_height, target_width, 2]`.
      Moreover, the offsets along the y-axis are rescaled by a factor equal to
      (target_height - 1) / (reference_height - 1) and the offsets along the
      x-axis are rescaled by a factor equal to
      (target_width - 1) / (reference_width - 1).
  """
  input_size_y = tf.shape(input_tensor)[1]
  input_size_x = tf.shape(input_tensor)[2]

  scale_y = tf.cast(target_size[0] - 1, tf.float32) / tf.cast(
      input_size_y - 1, tf.float32)
  scale_x = tf.cast(target_size[1] - 1, tf.float32) / tf.cast(
      input_size_x - 1, tf.float32)

  target_y, target_x = tf.split(
      value=input_tensor, num_or_size_splits=2, axis=3)
  target_y *= scale_y
  target_x *= scale_x
  target = tf.concat([target_y, target_x], 3)
  return resize_bilinear(target, target_size)


def resize_align_corners(input_tensor, target_size, method='bilinear'):
  """Resizes the input_tensor to target_size.

  This returns the same output as tf.compat.v1.image.resize(input_tensor,
  target_size, align_corners=True).

  Args:
    input_tensor: A tf.Tensor of shape [batch, height, width, channels].
    target_size: A list or tuple or 1D tf.Tensor that specifies the height and
      width after resizing.
    method: An optional string specifying the method used for resizing.
      Supported options are 'nearest' and 'bilinear'.

  Returns:
    The resized tensor.

  Raises:
    ValueError: An error occurs if 1) the input tensor's rank is not 4 or 2) the
      resizing method is not supported.
  """
  if method == 'bilinear':
    tf_method = tf.compat.v1.image.ResizeMethod.BILINEAR
  elif method == 'nearest':
    tf_method = tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR
  else:
    raise ValueError('The given method %s is not supported. Please use bilinear'
                     ' or nearest.' % method)

  tf.debugging.assert_rank(
      input_tensor, 4,
      message='Input tensor to resize method should have rank of 4.')

  return tf.compat.v1.image.resize(
      input_tensor,
      target_size,
      method=tf_method,
      align_corners=True,
      name='resize_align_corners')


def resize_bilinear(images,
                    size,
                    align_corners=True,
                    name=None):
  """TPU memory efficient version of tf.compat.v1.image.resize_bilinear.

  ResizeBilinear on TPU requires padded batch and channel dimensions. On a
  TPUv3, the worst case could lead to 256x memory consumption, if the
  input is, for example, [1, 257, 513, 1]. In this function, we replace the
  default resize_bilinear by two resize_bilinear operations, which put one image
  axis on the channel axis. This reduces TPU padding when batch * channel is
  small and height * width is large.

  Args:
    images: Input image of shape [B, H, W, C].
    size: A list of two elements: [height, width]. The new size for the images.
    align_corners: Whether to align corners of the image.
    name: Name of the operation.

  Returns:
    Resized image.
  """
  _, height, width, channel = images.get_shape().as_list()
  if height == size[0] and width == size[1]:
    return images
  dtype = images.dtype
  images = tf.cast(images, tf.float32)
  # We check the channel axis only since the batch size is similar (usually 1 or
  # 2). In this way, this if-else easily supports dynamic batch size without
  # using tf.cond().
  if channel > 32 or not align_corners:
    images = tf.compat.v1.image.resize_bilinear(
        images, size,
        align_corners=align_corners,
        name=name)
  else:
    images = tf.transpose(images, [0, 3, 1, 2])
    images = tf.compat.v1.image.resize_bilinear(
        images, [channel, size[0]],
        align_corners=align_corners,
        name=name + '_height' if name else None)
    images = tf.transpose(images, [0, 1, 3, 2])
    images = tf.compat.v1.image.resize_bilinear(
        images, [channel, size[1]],
        align_corners=align_corners,
        name=name + '_width' if name else None)
    images = tf.transpose(images, [0, 3, 2, 1])
  return tf.cast(images, dtype)


def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None) -> int:
  """Ensures all layers have channels that are divisible by the divisor.

  Args:
    value: A `float` of original value.
    divisor: An `int` of the divisor that needs to be checked upon.
    min_value: A `float` of  minimum value threshold.

  Returns:
    The adjusted value in `int` that is divisible by divisor.

  Raises:
    ValueError: Minimual value should be divisible by divisor.
  """
  if min_value is None:
    min_value = divisor
  elif min_value % divisor != 0:
    raise ValueError('Minimual value should be divisible by divisor.')

  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_value < 0.9 * value:
    new_value += divisor
  return int(new_value)


def transpose_and_reshape_for_attention_operation(inputs):
  """Sequentially transposes and reshapes the tensor.

  Args:
    inputs: An input [batch, num_heads, length, channel] tensor.

  Returns:
    output: An output [batch, length, num_heads * channel] tensor.
  """
  _, num_heads, length, channel = inputs.get_shape().as_list()
  transposed_inputs = tf.transpose(inputs, [0, 2, 1, 3])
  return tf.reshape(transposed_inputs, [-1, length, num_heads * channel])


def reshape_and_transpose_for_attention_operation(inputs, num_heads):
  """Sequentially reshapes and transposes the tensor.

  Args:
    inputs: An input [batch, length, num_heads * channel] tensor.
    num_heads: An integer, the number of attention heads.

  Returns:
    output: An output [batch, num_heads, length, channel] tensor.
  """
  _, length, channels = inputs.get_shape().as_list()
  inputs = tf.reshape(inputs, [-1, length, num_heads, channels // num_heads])
  return tf.transpose(inputs, [0, 2, 1, 3])


def get_layer_name(private_attribute_name):
  if private_attribute_name[0] != '_':
    raise ValueError('Private attribute name should start with a \'_\'.')
  return private_attribute_name[1:]


def get_stem_current_name(index):
  return '_basic_block{}'.format(index + 1)


def get_low_level_conv_fusion_conv_current_names(index):
  return ('_low_level_conv{}'.format(index + 1),
          '_fusion_conv{}'.format(index + 1))


def get_conv_bn_act_current_name(index, use_bn, activation):
  name = '_conv{}'.format(index + 1)
  if use_bn:
    name += '_bn'
  if (activation is not None and
      activation.lower() != 'none' and
      activation.lower() != 'linear'):
    name += '_act'
  return name


def safe_setattr(obj, name, value):
  """A conflict-safe version of setattr().

  Different from setattr(), this function raises ValueError if the object
  already has an attribute with the same name.

  Args:
    obj: An object whose attribute has to be set.
    name: A string, the name of the attribute.
    value: Any type, the value given to the attribute.

  Raises:
    ValueError: If the object already has an attribute with the same name.
  """
  if hasattr(obj, name):
    raise ValueError('The object already has an attribute with the same name.')
  setattr(obj, name, value)
