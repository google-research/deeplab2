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

"""Utility functions to set up unit tests on Panoptic Segmentation code."""

import os
from typing import Mapping, Optional, Tuple

from absl import flags
import numpy as np
from PIL import Image

import tensorflow as tf

FLAGS = flags.FLAGS

_TEST_DATA_DIR = ('deeplab2/'
                  'evaluation/testdata')


def read_test_image(testdata_path: str,
                    image_format: Optional[str] = None) -> np.ndarray:
  """Loads a test image.

  Args:
    testdata_path: Image path relative to panoptic_segmentation/testdata as a
      string.
    image_format: Format of the image. Can be one of 'RGBA', 'RGB', or 'L'.

  Returns:
    The image, as a numpy array.
  """
  image_path = os.path.join(_TEST_DATA_DIR, testdata_path)
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    image = Image.open(f)
    if image_format is not None:
      image = image.convert(image_format)
    return np.array(image)


def read_segmentation_with_rgb_color_map(
    image_testdata_path: str,
    rgb_to_semantic_label: Mapping[Tuple[int, int, int], int],
    output_dtype: Optional[np.dtype] = None) -> np.ndarray:
  """Reads a test segmentation as an image and a map from colors to labels.

  Args:
    image_testdata_path: Image path relative to panoptic_segmentation/testdata
      as a string.
    rgb_to_semantic_label: Mapping from RGB colors to integer labels as a
      dictionary.
    output_dtype: Type of the output labels. If None, defaults to the type of
      the provided color map.

  Returns:
    A 2D numpy array of labels.

  Raises:
    ValueError: On an incomplete `rgb_to_semantic_label`.
  """
  rgb_image = read_test_image(image_testdata_path, image_format='RGB')
  if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
    raise AssertionError('Expected RGB image, actual shape is %s' %
                         (rgb_image.shape,))

  num_pixels = rgb_image.shape[0] * rgb_image.shape[1]
  unique_colors = np.unique(np.reshape(rgb_image, [num_pixels, 3]), axis=0)
  if not set(map(tuple, unique_colors)).issubset(rgb_to_semantic_label.keys()):
    raise ValueError('RGB image has colors not in color map.')

  output_dtype = output_dtype or type(
      next(iter(rgb_to_semantic_label.values())))
  output_labels = np.empty(rgb_image.shape[:2], dtype=output_dtype)
  for rgb_color, int_label in rgb_to_semantic_label.items():
    color_array = np.array(rgb_color, ndmin=3)
    output_labels[np.all(rgb_image == color_array, axis=2)] = int_label
  return output_labels


def panoptic_segmentation_with_class_map(
    instance_testdata_path: str, instance_label_to_semantic_label: Mapping[int,
                                                                           int]
) -> Tuple[np.ndarray, np.ndarray]:
  """Reads in a panoptic segmentation with an instance map and a map to classes.

  Args:
    instance_testdata_path: Path to a grayscale instance map, given as a string
      and relative to panoptic_segmentation/testdata.
    instance_label_to_semantic_label: A map from instance labels to class
      labels.

  Returns:
    A tuple `(instance_labels, class_labels)` of numpy arrays.

  Raises:
    ValueError: On a mismatched set of instances in
    the
      `instance_label_to_semantic_label`.
  """
  instance_labels = read_test_image(instance_testdata_path, image_format='L')
  if set(np.unique(instance_labels)) != set(
      instance_label_to_semantic_label.keys()):
    raise ValueError('Provided class map does not match present instance ids.')

  class_labels = np.empty_like(instance_labels)
  for instance_id, class_id in instance_label_to_semantic_label.items():
    class_labels[instance_labels == instance_id] = class_id

  return instance_labels, class_labels
