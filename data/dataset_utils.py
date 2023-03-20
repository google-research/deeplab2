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

"""This file contains utility function for handling the dataset."""

import tensorflow as tf


def get_semantic_and_panoptic_label(dataset_info, label, ignore_label):
  """Helper function to get semantic and panoptic label from panoptic label.

  This functions gets the semantic and panoptic label from panoptic label for
  different datasets. The labels must be encoded with semantic_label *
  label_divisor + instance_id. For thing classes, the instance ID 0 is reserved
  for crowd regions. Please note, the returned panoptic label has replaced
  the crowd region with ignore regions. Yet, the semantic label makes use of
  these regions.

  Args:
    dataset_info: A dictionary storing dataset information.
    label: A Tensor of panoptic label.
    ignore_label: An integer specifying the ignore_label.

  Returns:
    semantic_label: A Tensor of semantic segmentation label.
    panoptic_label: A Tensor of panoptic segmentation label, which follows the
      Cityscapes annotation where
      panoptic_label = semantic_label * panoptic_label_divisor + instance_id.
    thing_mask: A boolean Tensor specifying the thing regions. Zero if no thing.
    crowd_region: A boolean Tensor specifying crowd region. Zero if no crowd
      annotation.

  Raises:
    ValueError: An error occurs when the ignore_label is not in range
      [0, label_divisor].
  """
  panoptic_label_divisor = dataset_info['panoptic_label_divisor']
  if ignore_label >= panoptic_label_divisor or ignore_label < 0:
    raise ValueError('The ignore_label must be in [0, label_divisor].')

  semantic_label = label // panoptic_label_divisor
  # Find iscrowd region if any and set to ignore for panoptic labels.
  # 1. Find thing mask.
  thing_mask = tf.zeros_like(semantic_label, tf.bool)
  for thing_id in dataset_info['class_has_instances_list']:
    thing_mask = tf.logical_or(
        thing_mask,
        tf.equal(semantic_label, thing_id))
  # 2. Find crowd region (thing label that have instance_id == 0).
  crowd_region = tf.logical_and(
      thing_mask,
      tf.equal(label % panoptic_label_divisor, 0))
  # 3. Set crowd region to ignore label.
  panoptic_label = tf.where(
      crowd_region,
      tf.ones_like(label) * ignore_label * panoptic_label_divisor,
      label)

  return semantic_label, panoptic_label, thing_mask, crowd_region
