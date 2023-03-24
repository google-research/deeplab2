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

"""Utility functions related to preprocessing inputs."""

from typing import Tuple, Sequence, Optional, Union

import numpy as np
import tensorflow as tf


def flip_dim(tensor_list: Sequence[tf.Tensor], prob: float = 0.5, dim: int = 1
             ) -> Sequence[tf.Tensor]:
  """Randomly flips a dimension of the given tensor.

  The decision to randomly flip the `Tensors` is made together. In other words,
  all or none of the images pass in are flipped.

  Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
  that we can control for the probability as well as ensure the same decision
  is applied across the images.

  Args:
    tensor_list: A list of `Tensors` with the same number of dimensions.
    prob: The probability of a left-right flip.
    dim: The dimension to flip, 0, 1, ..

  Returns:
    outputs: A list of the possibly flipped `Tensors` as well as an indicator
    `Tensor` at the end whose value is `True` if the inputs were flipped and
    `False` otherwise.

  Raises:
    ValueError: If dim is negative or greater than the dimension of a `Tensor`.
  """
  random_value = tf.random.uniform([])

  def flip():
    flipped = []
    for tensor in tensor_list:
      if dim < 0 or dim >= len(tensor.get_shape().as_list()):
        raise ValueError('dim must represent a valid dimension.')
      flipped.append(tf.reverse(tensor, [dim]))
    return flipped

  is_flipped = tf.less_equal(random_value, prob)
  outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
  if not isinstance(outputs, (list, tuple)):
    outputs = [outputs]
  outputs.append(is_flipped)

  return outputs


def get_label_resize_method(label: tf.Tensor) -> tf.image.ResizeMethod:
  """Returns the resize method of labels depending on label dtype.

  Args:
    label: Groundtruth label tensor.

  Returns:
    tf.image.ResizeMethod.BILINEAR, if label dtype is floating.
    tf.image.ResizeMethod.NEAREST_NEIGHBOR, if label dtype is integer.

  Raises:
    ValueError: If label is neither floating nor integer.
  """
  if label.dtype.is_floating:
    return tf.image.ResizeMethod.BILINEAR
  elif label.dtype.is_integer:
    return tf.image.ResizeMethod.NEAREST_NEIGHBOR
  else:
    raise ValueError('Label type must be either floating or integer.')


def crop(image: tf.Tensor,
         offset_height: int,
         offset_width: int,
         crop_height: int,
         crop_width: int) -> tf.Tensor:
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    The cropped (and resized) image.

  Raises:
    ValueError: if `image` doesn't have rank of 3.
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  if len(image.get_shape().as_list()) != 3:
    raise ValueError('input must have rank of 3')
  original_channels = image.get_shape().as_list()[2]

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  image = tf.reshape(image, cropped_shape)
  image.set_shape([crop_height, crop_width, original_channels])
  return image


def random_crop(image_list: Sequence[tf.Tensor],
                crop_height: int,
                crop_width: int) -> Sequence[tf.Tensor]:
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3), [
            'Wrong rank for tensor %d in image_list [expected] [actual]', i, 3,
            image_rank
        ])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height), [
            'Wrong height for tensor %d in image_list [expected][actual]', i,
            height, image_height
        ])
    width_assert = tf.Assert(
        tf.equal(width, image_width), [
            'Wrong width for tensor %d in image_list [expected][actual]', i,
            width, image_width
        ])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random.uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random.uniform([],
                                    maxval=max_offset_height,
                                    dtype=tf.int32)
  offset_width = tf.random.uniform([], maxval=max_offset_width, dtype=tf.int32)

  return [crop(image, offset_height, offset_width,
               crop_height, crop_width) for image in image_list]


def get_random_scale(min_scale_factor: float,
                     max_scale_factor: float,
                     step_size: float) -> tf.Tensor:
  """Gets a random scale value.

  Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.

  Returns:
    A tensor with random scale value selected between minimum and maximum value.
    If `min_scale_factor` and `max_scale_factor` are the same, a number is
    returned instead.

  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
  if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
    raise ValueError('Unexpected value of min_scale_factor.')

  if min_scale_factor == max_scale_factor:
    return np.float32(min_scale_factor)  # pytype: disable=bad-return-type  # numpy-scalars

  # When step_size = 0, we sample the value uniformly from [min, max).
  if step_size == 0:
    return tf.random.uniform([1],
                             minval=min_scale_factor,
                             maxval=max_scale_factor)

  # When step_size != 0, we randomly select one discrete value from [min, max].
  num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
  scale_factors = tf.linspace(min_scale_factor, max_scale_factor, num_steps)
  shuffled_scale_factors = tf.random.shuffle(scale_factors)
  return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image: tf.Tensor,
                                   label: Optional[tf.Tensor] = None,
                                   scale: float = 1.0
                                   ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
  """Randomly scales image and label.

  Args:
    image: Image with shape [height, width, 3].
    label: Label with shape [height, width, 1].
    scale: The value to scale image and label.

  Returns:
    Scaled image and label.
  """
  # No random scaling if scale == 1.
  if scale == 1.0:
    return image, label
  image_shape = tf.shape(image)
  new_dim = tf.cast(
      tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale,
      tf.int32)

  # Need squeeze and expand_dims because image interpolation takes
  # 4D tensors as input.
  image = tf.squeeze(
      tf.compat.v1.image.resize_bilinear(
          tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
  if label is not None:
    label = tf.compat.v1.image.resize(
        label,
        new_dim,
        method=get_label_resize_method(label),
        align_corners=True)

  return image, label


def resolve_shape(tensor: tf.Tensor, rank: Optional[int] = None) -> tf.Tensor:
  """Fully resolves the shape of a Tensor.

  Use as much as possible the shape components already known during graph
  creation and resolve the remaining ones during runtime.

  Args:
    tensor: Input tensor whose shape we query.
    rank: The rank of the tensor, provided that we know it.

  Returns:
    shape: The full shape of the tensor.
  """
  if rank is not None:
    shape = tensor.get_shape().with_rank(rank).as_list()
  else:
    shape = tensor.get_shape().as_list()

  if None in shape:
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):
      if shape[i] is None:
        shape[i] = dynamic_shape[i]

  return shape


def _scale_dim(original_size: int, factor: float) -> int:
  """Helper method to scale one input dimension by the given factor."""
  original_size = tf.cast(original_size, tf.float32)
  factor = tf.cast(factor, tf.float32)
  return tf.cast(tf.floor(original_size * factor), tf.int32)


def process_resize_value(resize_spec: Optional[Union[int, Tuple[int, int]]]
                         ) -> Optional[Tuple[int, int]]:
  """Helper method to process input resize spec.

  Args:
    resize_spec: Either None, a python scalar, or a sequence with length <=2.
      Each value in the sequence should be a python integer.

  Returns:
    None if input size is not valid, or 2-tuple of (height, width), derived
      from input resize_spec.
  """
  if not resize_spec:
    return None

  if isinstance(resize_spec, int):
    # For conveniences and also backward compatibility.
    resize_spec = (resize_spec,)

  resize_spec = tuple(resize_spec)

  if len(resize_spec) == 1:
    resize_spec = (resize_spec[0], resize_spec[0])

  if len(resize_spec) != 2:
    raise ValueError('Unable to process input resize_spec: %s' % resize_spec)

  if resize_spec[0] <= 0 or resize_spec[1] <= 0:
    return None

  return resize_spec


def _resize_to_match_min_size(input_shape: Tuple[int, int],
                              min_size: Tuple[int, int]) -> Tuple[int, int]:
  """Returns the resized shape so that both sides match minimum size.

  Note: the input image will still be scaled if input height and width
  are already greater than minimum size.

  Args:
    input_shape: A 2-tuple, (height, width) of the input image. Each value can
      be either a python integer or a integer scalar tensor.
    min_size: A tuple of (minimum height, minimum width) to specify the
      minimum shape after resize. The input shape would be scaled so that both
      height and width will be greater than or equal to their minimum value.

  Returns:
    A 2-tuple, (height, width), resized input shape which preserves input
      aspect ratio.
  """
  input_height, input_width = input_shape
  min_height, min_width = min_size

  scale_factor = tf.maximum(min_height / input_height, min_width / input_width)
  return (_scale_dim(input_height, scale_factor),
          _scale_dim(input_width, scale_factor))


def _resize_to_fit_max_size(input_shape: Tuple[int, int],
                            max_size: Tuple[int, int]) -> Tuple[int, int]:
  """Returns the resized shape so that both sides fit within max size.

  Note: if input shape is already smaller or equal to maximum size, no resize
    operation would be performed.

  Args:
    input_shape: A 2-tuple, (height, width) of the input image. Each value can
      be either a python integer or a integer scalar tensor.
    max_size: A tuple of (minimum height, minimum width) to specify
      the maximum allowed shape after resize.

  Returns:
    A 2-tuple, (height, width), resized input shape which preserves input
      aspect ratio.
  """
  input_height, input_width = input_shape
  max_height, max_width = max_size
  scale_factor = tf.minimum(max_height / input_height, max_width / input_width)

  scale_factor = tf.minimum(tf.cast(scale_factor, tf.float32),
                            tf.cast(1.0, tf.float32))
  return (_scale_dim(input_height, scale_factor),
          _scale_dim(input_width, scale_factor))


def resize_to_range_helper(input_shape: Tuple[int, int],
                           min_size: Tuple[int, int],
                           max_size: Optional[Tuple[int, int]] = None,
                           factor: Optional[int] = None) -> tf.Tensor:
  """Determines output size in specified range.

  The output size (height and/or width) can be described by two cases:
  1. If current side can be rescaled so its minimum size is equal to min_size
     without the other side exceeding its max_size, then do so.
  2. Otherwise, resize so at least one side is reaching its max_size.

  An integer in `range(factor)` is added to the computed sides so that the
  final dimensions are multiples of `factor` plus one.

  Args:
    input_shape: A 2-tuple, (height, width) of the input image. Each value can
      be either a python integer or a integer scalar tensor.
    min_size: A 2-tuple of (height, width), desired minimum value after resize.
      If a single element is given, then height and width share the same
      min_size. None, empty or having 0 indicates no minimum value will be used.
    max_size: A 2-tuple of (height, width), maximum allowed value after resize.
      If a single element is given, then height and width share the same
      max_size. None, empty or having 0 indicates no maximum value will be used.
      Note that the output dimension is no larger than max_size and may be
      slightly smaller than max_size when factor is not None.
    factor: None or integer, make output size multiple of factor plus one.

  Returns:
    A 1-D tensor containing the [new_height, new_width].
  """
  output_shape = input_shape

  min_size = process_resize_value(min_size)
  if min_size:
    output_shape = _resize_to_match_min_size(input_shape, min_size)

  max_size = process_resize_value(max_size)
  if max_size:
    if factor:
      # Update max_size to be a multiple of factor plus 1 and make sure the
      # max dimension after resizing is no larger than max_size.
      max_size = (max_size[0] - (max_size[0] - 1) % factor,
                  max_size[1] - (max_size[1] - 1) % factor)

    output_shape = _resize_to_fit_max_size(output_shape, max_size)

  output_shape = tf.stack(output_shape)
  # Ensure that both output sides are multiples of factor plus one.
  if factor:
    output_shape += (factor - (output_shape - 1) % factor) % factor

  return output_shape


def resize_to_range(
    image: tf.Tensor,
    label: Optional[tf.Tensor] = None,
    min_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None,
    factor: Optional[int] = None,
    align_corners: bool = True,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
  """Resizes image or label so their sides are within the provided range.

  The output size (height and/or width) can be described by two cases:
  1. If current side can be rescaled so its minimum size is equal to min_size
     without the other side exceeding its max_size, then do so.
  2. Otherwise, resize so at least one side is reaching its max_size.

  An integer in `range(factor)` is added to the computed sides so that the
  final dimensions are multiples of `factor` plus one.

  Args:
    image: A 3D tensor of shape [height, width, channels].
    label: (optional) A 3D tensor of shape [height, width, channels].
    min_size: A 2-tuple of (height, width), desired minimum value after resize.
      If a single element is given, then height and width share the same
      min_size. None, empty or having 0 indicates no minimum value will be used.
    max_size: A 2-tuple of (height, width), maximum allowed value after resize.
      If a single element is given, then height and width share the same
      max_size. None, empty or having 0 indicates no maximum value will be used.
      Note that the output dimension is no larger than max_size and may be
      slightly smaller than max_size when factor is not None.
    factor: Make output size multiple of factor plus one.
    align_corners: If True, exactly align all 4 corners of input and output.
    method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

  Returns:
    resized_image: A 3-D tensor of shape [new_height, new_width, channels],
      where the image has been resized with the specified method.
    resized_label: Either None (if input label is None) or a 3-D tensor,
      where the input label has been resized accordingly.

  Raises:
    ValueError: If the image is not a 3D tensor.
  """
  orig_height, orig_width, _ = resolve_shape(image, rank=3)
  new_size = resize_to_range_helper(input_shape=(orig_height, orig_width),
                                    min_size=min_size,
                                    max_size=max_size,
                                    factor=factor)

  resized_image = tf.compat.v1.image.resize(
      image, new_size, method=method, align_corners=align_corners)

  if label is None:
    return resized_image, None

  resized_label = tf.compat.v1.image.resize(
      label,
      new_size,
      method=get_label_resize_method(label),
      align_corners=align_corners)

  return resized_image, resized_label
