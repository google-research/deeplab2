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

"""Common utility functions and classes for building multicamera dataset."""

import tensorflow as tf

from deeplab2 import common


class MultiCameraSegmentationDecoder(object):
  """Multiview parser to decode serialized tf.Example.

  The decoded dict contains information for all cameras, which are in
  `camera_name: camera_value` format. `camera_value` is in the format of normal
  datasets defined in `SegmentationDecoder`.
  """

  def __init__(self,
               is_panoptic_dataset=True,
               is_video_dataset=False,
               is_depth_dataset=False,
               cameras_to_use=None,
               use_next_frame=False,
               decode_groundtruth_label=True):
    self._is_panoptic_dataset = is_panoptic_dataset
    self._is_video_dataset = is_video_dataset
    self._is_depth_dataset = is_depth_dataset
    self._cameras_to_use = cameras_to_use
    self._use_next_frame = use_next_frame
    self._decode_groundtruth_label = decode_groundtruth_label

    string_feature = tf.io.FixedLenFeature((), tf.string)
    int_feature = tf.io.FixedLenFeature((), tf.int64)
    self._keys_to_features = {
        common.KEY_IMAGE_FILENAME: string_feature,
        common.KEY_IMAGE_FORMAT: string_feature,
        common.KEY_IMAGE_CHANNELS: int_feature,
    }
    for view in cameras_to_use:
      self._keys_to_features[common.KEY_PER_CAMERA_ENCODED_IMAGE %
                             view] = string_feature
      self._keys_to_features[common.KEY_PER_CAMERA_IMAGE_HEIGHT %
                             view] = int_feature
      self._keys_to_features[common.KEY_PER_CAMERA_IMAGE_WIDTH %
                             view] = int_feature
      if decode_groundtruth_label:
        self._keys_to_features[common.KEY_PER_CAMERA_ENCODED_LABEL %
                               view] = string_feature
      # Next-frame specific processing.
      if self._use_next_frame:
        self._keys_to_features[common.KEY_PER_CAMERA_ENCODED_NEXT_IMAGE %
                               view] = string_feature
        if decode_groundtruth_label:
          self._keys_to_features[common.KEY_PER_CAMERA_ENCODED_NEXT_LABEL %
                                 view] = string_feature
      # Depth specific processing.
      if self._is_depth_dataset and decode_groundtruth_label:
        self._keys_to_features[common.KEY_PER_CAMERA_ENCODED_DEPTH %
                               view] = string_feature
    if self._is_video_dataset:
      self._keys_to_features[common.KEY_SEQUENCE_ID] = string_feature
      self._keys_to_features[common.KEY_FRAME_ID] = string_feature

  def _decode_image(self, parsed_tensors, key):
    """Decodes image under key from parsed tensors."""
    image = tf.io.decode_image(
        parsed_tensors[key],
        channels=3,
        dtype=tf.dtypes.uint8,
        expand_animations=False)
    image.set_shape([None, None, 3])
    return image

  def _decode_label(self,
                    parsed_tensors,
                    label_key_template,
                    view,
                    out_type=tf.int32):
    """Decodes segmentation label under label_key from parsed tensors."""
    if self._is_panoptic_dataset:
      flattened_label = tf.io.decode_raw(
          parsed_tensors[label_key_template % view], out_type=out_type)
      label_shape = tf.stack([
          parsed_tensors[common.KEY_PER_CAMERA_IMAGE_HEIGHT % view],
          parsed_tensors[common.KEY_PER_CAMERA_IMAGE_WIDTH % view], 1
      ])
      label = tf.reshape(flattened_label, label_shape)
      return label

    label = tf.io.decode_image(parsed_tensors[label_key_template], channels=1)
    label.set_shape([None, None, 1])
    return label

  def __call__(self, serialized_example):
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, features=self._keys_to_features)
    camera_dict_per_view = {}
    for view in self._cameras_to_use:
      view_dict = {
          'image_name':
              parsed_tensors[common.KEY_IMAGE_FILENAME],
          'image':
              self._decode_image(parsed_tensors,
                                 common.KEY_PER_CAMERA_ENCODED_IMAGE % view),
          'height':
              tf.cast(
                  parsed_tensors[common.KEY_PER_CAMERA_IMAGE_HEIGHT % view],
                  dtype=tf.int32),
          'width':
              tf.cast(
                  parsed_tensors[common.KEY_PER_CAMERA_IMAGE_WIDTH % view],
                  dtype=tf.int32),
      }
      view_dict['label'] = None
      if self._decode_groundtruth_label:
        view_dict['label'] = self._decode_label(
            parsed_tensors, common.KEY_PER_CAMERA_ENCODED_LABEL, view)
      if self._is_video_dataset:
        view_dict['sequence'] = parsed_tensors[common.KEY_SEQUENCE_ID]
        view_dict['frame_id'] = parsed_tensors[common.KEY_FRAME_ID]
      if self._use_next_frame:
        view_dict['next_image'] = self._decode_image(
            parsed_tensors, common.KEY_PER_CAMERA_ENCODED_NEXT_IMAGE % view)
        if self._decode_groundtruth_label:
          view_dict['next_label'] = self._decode_label(
              parsed_tensors, common.KEY_PER_CAMERA_ENCODED_NEXT_LABEL, view)
      if self._is_depth_dataset and self._decode_groundtruth_label:
        view_dict['depth'] = self._decode_label(
            parsed_tensors, common.KEY_PER_CAMERA_ENCODED_DEPTH, view)

      camera_dict_per_view[view] = view_dict
    return camera_dict_per_view
