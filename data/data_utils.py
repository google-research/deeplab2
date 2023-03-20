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

"""Contains common utility functions and classes for building dataset."""

from collections import abc
import io

import numpy as np
from PIL import Image
from PIL import ImageOps
import tensorflow as tf

from deeplab2 import common

_PANOPTIC_LABEL_FORMAT = 'raw'


def read_image(image_data):
  """Decodes image from in-memory data.

  Args:
    image_data: Bytes data representing encoded image.

  Returns:
    Decoded PIL.Image object.
  """
  image = Image.open(io.BytesIO(image_data))

  try:
    image = ImageOps.exif_transpose(image)
  except TypeError:
    # capture and ignore this bug:
    # https://github.com/python-pillow/Pillow/issues/3973
    pass

  return image


def get_image_dims(image_data, check_is_rgb=False):
  """Decodes image and return its height and width.

  Args:
    image_data: Bytes data representing encoded image.
    check_is_rgb: Whether to check encoded image is RGB.

  Returns:
    Decoded image size as a tuple of (height, width)

  Raises:
    ValueError: If check_is_rgb is set and input image has other format.
  """
  image = read_image(image_data)

  if check_is_rgb and image.mode != 'RGB':
    raise ValueError('Expects RGB image data, gets mode: %s' % image.mode)

  width, height = image.size
  return height, width


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or an iterable of integer values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, abc.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  if isinstance(values, str):
    values = values.encode()

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def create_features(image_data,
                    image_format,
                    filename,
                    label_data=None,
                    label_format=None):
  """Creates image/segmentation features.

  Args:
    image_data: String or byte stream of encoded image data.
    image_format: String, image data format, should be either 'jpeg', 'jpg', or
      'png'.
    filename: String, image filename.
    label_data: String or byte stream of (potentially) encoded label data. If
      None, we skip to write it to tf.train.Example.
    label_format: String, label data format, should be either 'png' or 'raw'. If
      None, we skip to write it to tf.train.Example.

  Returns:
    A dictionary of feature name to tf.train.Feature maaping.
  """
  if image_format not in ('jpeg', 'png', 'jpg'):
    raise ValueError('Unsupported image format: %s' % image_format)

  # Check color mode, and convert grey image to rgb image.
  image = read_image(image_data)
  if image.mode != 'RGB':
    image = image.convert('RGB')
    image_data = io.BytesIO()
    image.save(image_data, format=image_format)
    image_data = image_data.getvalue()

  height, width = get_image_dims(image_data, check_is_rgb=True)

  feature_dict = {
      common.KEY_ENCODED_IMAGE: _bytes_list_feature(image_data),
      common.KEY_IMAGE_FILENAME: _bytes_list_feature(filename),
      common.KEY_IMAGE_FORMAT: _bytes_list_feature(image_format),
      common.KEY_IMAGE_HEIGHT: _int64_list_feature(height),
      common.KEY_IMAGE_WIDTH: _int64_list_feature(width),
      common.KEY_IMAGE_CHANNELS: _int64_list_feature(3),
  }

  if label_data is None:
    return feature_dict

  if label_format == 'png':
    label_height, label_width = get_image_dims(label_data)
    if (label_height, label_width) != (height, width):
      raise ValueError('Image (%s) and label (%s) shape mismatch' %
                       ((height, width), (label_height, label_width)))
  elif label_format == 'raw':
    # Raw label encodes int32 array.
    expected_label_size = height * width * np.dtype(np.int32).itemsize
    if len(label_data) != expected_label_size:
      raise ValueError('Expects raw label data length %d, gets %d' %
                       (expected_label_size, len(label_data)))
  else:
    raise ValueError('Unsupported label format: %s' % label_format)

  feature_dict.update({
      common.KEY_ENCODED_LABEL: _bytes_list_feature(label_data),
      common.KEY_LABEL_FORMAT: _bytes_list_feature(label_format)
  })

  return feature_dict


def create_tfexample(image_data,
                     image_format,
                     filename,
                     label_data=None,
                     label_format=None):
  """Converts one image/segmentation pair to TF example.

  Args:
    image_data: String or byte stream of encoded image data.
    image_format: String, image data format, should be either 'jpeg' or 'png'.
    filename: String, image filename.
    label_data: String or byte stream of (potentially) encoded label data. If
      None, we skip to write it to tf.train.Example.
    label_format: String, label data format, should be either 'png' or 'raw'. If
      None, we skip to write it to tf.train.Example.

  Returns:
    TF example proto.
  """
  feature_dict = create_features(image_data, image_format, filename, label_data,
                                 label_format)
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def create_video_tfexample(image_data,
                           image_format,
                           filename,
                           sequence_id,
                           image_id,
                           label_data=None,
                           label_format=None,
                           prev_image_data=None,
                           prev_label_data=None):
  """Converts one video frame/panoptic segmentation pair to TF example.

  Args:
    image_data: String or byte stream of encoded image data.
    image_format: String, image data format, should be either 'jpeg' or 'png'.
    filename: String, image filename.
    sequence_id: ID of the video sequence as a string.
    image_id: ID of the image as a string.
    label_data: String or byte stream of (potentially) encoded label data. If
      None, we skip to write it to tf.train.Example.
    label_format: String, label data format, should be either 'png' or 'raw'. If
      None, we skip to write it to tf.train.Example.
    prev_image_data: An optional string or byte stream of encoded previous image
      data.
    prev_label_data: An optional string or byte stream of (potentially) encoded
      previous label data.

  Returns:
    TF example proto.
  """
  feature_dict = create_features(image_data, image_format, filename, label_data,
                                 label_format)
  feature_dict.update({
      common.KEY_SEQUENCE_ID: _bytes_list_feature(sequence_id),
      common.KEY_FRAME_ID: _bytes_list_feature(image_id)
  })
  if prev_image_data is not None:
    feature_dict[common.KEY_ENCODED_PREV_IMAGE] = _bytes_list_feature(
        prev_image_data)
  if prev_label_data is not None:
    feature_dict[common.KEY_ENCODED_PREV_LABEL] = _bytes_list_feature(
        prev_label_data)
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def create_video_and_depth_tfexample(image_data,
                                     image_format,
                                     filename,
                                     sequence_id,
                                     image_id,
                                     label_data=None,
                                     label_format=None,
                                     next_image_data=None,
                                     next_label_data=None,
                                     depth_data=None,
                                     depth_format=None):
  """Converts an image/segmentation pair and depth of first frame to TF example.

    The image pair contains the current frame and the next frame with the
    current frame including depth label.

  Args:
    image_data: String or byte stream of encoded image data.
    image_format: String, image data format, should be either 'jpeg' or 'png'.
    filename: String, image filename.
    sequence_id: ID of the video sequence as a string.
    image_id: ID of the image as a string.
    label_data: String or byte stream of (potentially) encoded label data. If
      None, we skip to write it to tf.train.Example.
    label_format: String, label data format, should be either 'png' or 'raw'. If
      None, we skip to write it to tf.train.Example.
    next_image_data: An optional string or byte stream of encoded next image
      data.
    next_label_data: An optional string or byte stream of (potentially) encoded
      next label data.
    depth_data: An optional string or byte sream of encoded depth data.
    depth_format: String, depth data format, should be either 'png' or 'raw'.

  Returns:
    TF example proto.
  """
  feature_dict = create_features(image_data, image_format, filename, label_data,
                                 label_format)
  feature_dict.update({
      common.KEY_SEQUENCE_ID: _bytes_list_feature(sequence_id),
      common.KEY_FRAME_ID: _bytes_list_feature(image_id)
  })
  if next_image_data is not None:
    feature_dict[common.KEY_ENCODED_NEXT_IMAGE] = _bytes_list_feature(
        next_image_data)
  if next_label_data is not None:
    feature_dict[common.KEY_ENCODED_NEXT_LABEL] = _bytes_list_feature(
        next_label_data)
  if depth_data is not None:
    feature_dict[common.KEY_ENCODED_DEPTH] = _bytes_list_feature(
        depth_data)
    feature_dict[common.KEY_DEPTH_FORMAT] = _bytes_list_feature(
        depth_format)
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class SegmentationDecoder(object):
  """Basic parser to decode serialized tf.Example."""

  def __init__(self,
               is_panoptic_dataset=True,
               is_video_dataset=False,
               is_depth_dataset=False,
               use_two_frames=False,
               use_next_frame=False,
               decode_groundtruth_label=True):
    self._is_panoptic_dataset = is_panoptic_dataset
    self._is_video_dataset = is_video_dataset
    self._is_depth_dataset = is_depth_dataset
    self._use_two_frames = use_two_frames
    self._use_next_frame = use_next_frame
    self._decode_groundtruth_label = decode_groundtruth_label
    string_feature = tf.io.FixedLenFeature((), tf.string)
    int_feature = tf.io.FixedLenFeature((), tf.int64)
    self._keys_to_features = {
        common.KEY_ENCODED_IMAGE: string_feature,
        common.KEY_IMAGE_FILENAME: string_feature,
        common.KEY_IMAGE_FORMAT: string_feature,
        common.KEY_IMAGE_HEIGHT: int_feature,
        common.KEY_IMAGE_WIDTH: int_feature,
        common.KEY_IMAGE_CHANNELS: int_feature,
    }
    if decode_groundtruth_label:
      self._keys_to_features[common.KEY_ENCODED_LABEL] = string_feature
    if self._is_video_dataset:
      self._keys_to_features[common.KEY_SEQUENCE_ID] = string_feature
      self._keys_to_features[common.KEY_FRAME_ID] = string_feature
    # Two-frame specific processing.
    if self._use_two_frames:
      self._keys_to_features[common.KEY_ENCODED_PREV_IMAGE] = string_feature
      if decode_groundtruth_label:
        self._keys_to_features[common.KEY_ENCODED_PREV_LABEL] = string_feature
    # Next-frame specific processing.
    if self._use_next_frame:
      self._keys_to_features[common.KEY_ENCODED_NEXT_IMAGE] = string_feature
      if decode_groundtruth_label:
        self._keys_to_features[common.KEY_ENCODED_NEXT_LABEL] = string_feature
    # Depth specific processing.
    if self._is_depth_dataset and decode_groundtruth_label:
      self._keys_to_features[common.KEY_ENCODED_DEPTH] = string_feature

  def _decode_image(self, parsed_tensors, key):
    """Decodes image under key from parsed tensors."""
    image = tf.io.decode_image(
        parsed_tensors[key],
        channels=3,
        dtype=tf.dtypes.uint8,
        expand_animations=False)
    image.set_shape([None, None, 3])
    return image

  def _decode_label(self, parsed_tensors, label_key):
    """Decodes segmentation label under label_key from parsed tensors."""
    if self._is_panoptic_dataset:
      flattened_label = tf.io.decode_raw(
          parsed_tensors[label_key], out_type=tf.int32)
      label_shape = tf.stack([
          parsed_tensors[common.KEY_IMAGE_HEIGHT],
          parsed_tensors[common.KEY_IMAGE_WIDTH], 1
      ])
      label = tf.reshape(flattened_label, label_shape)
      return label

    label = tf.io.decode_image(parsed_tensors[label_key], channels=1)
    label.set_shape([None, None, 1])
    return label

  def __call__(self, serialized_example):
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, features=self._keys_to_features)
    return_dict = {
        'image':
            self._decode_image(parsed_tensors, common.KEY_ENCODED_IMAGE),
        'image_name':
            parsed_tensors[common.KEY_IMAGE_FILENAME],
        'height':
            tf.cast(parsed_tensors[common.KEY_IMAGE_HEIGHT], dtype=tf.int32),
        'width':
            tf.cast(parsed_tensors[common.KEY_IMAGE_WIDTH], dtype=tf.int32),
    }
    return_dict['label'] = None
    if self._decode_groundtruth_label:
      return_dict['label'] = self._decode_label(parsed_tensors,
                                                common.KEY_ENCODED_LABEL)
    if self._is_video_dataset:
      return_dict['sequence'] = parsed_tensors[common.KEY_SEQUENCE_ID]
    if self._use_two_frames:
      return_dict['prev_image'] = self._decode_image(
          parsed_tensors, common.KEY_ENCODED_PREV_IMAGE)
      if self._decode_groundtruth_label:
        return_dict['prev_label'] = self._decode_label(
            parsed_tensors, common.KEY_ENCODED_PREV_LABEL)
    if self._use_next_frame:
      return_dict['next_image'] = self._decode_image(
          parsed_tensors, common.KEY_ENCODED_NEXT_IMAGE)
      if self._decode_groundtruth_label:
        return_dict['next_label'] = self._decode_label(
            parsed_tensors, common.KEY_ENCODED_NEXT_LABEL)
    if self._is_depth_dataset and self._decode_groundtruth_label:
      return_dict['depth'] = self._decode_label(
          parsed_tensors, common.KEY_ENCODED_DEPTH)
    return return_dict
