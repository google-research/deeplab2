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

"""Script to generate test data for cityscapes."""

import collections
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# resources dependency

from deeplab2.data import data_utils
from deeplab2.data import dataset

flags.DEFINE_string(
    'panoptic_annotation_path',
    'deeplab2/data/testdata/'
    'dummy_prediction.png',
    'Path to annotated test image with cityscapes encoding.')
flags.DEFINE_string(
    'panoptic_gt_output_path',
    'deeplab2/data/testdata/'
    'dummy_gt_for_vps.png',
    'Path to annotated test image with Video Panoptic Segmentation encoding.')
flags.DEFINE_string(
    'output_cityscapes_root',
    'deeplab2/data/testdata/',
    'Path to output root directory.')

FLAGS = flags.FLAGS

# Cityscapes label, using `TrainId`.
_CITYSCAPES_IGNORE = 255
# Each valid (not ignored) label below is a tuple of (TrainId, EvalId)
_CITYSCAPES_CAR = (13, 26)
_CITYSCAPES_TREE = (8, 21)
_CITYSCAPES_SKY = (10, 23)
_CITYSCAPES_BUILDING = (2, 11)
_CITYSCAPES_ROAD = (0, 7)

_IS_CROWD = 'is_crowd'
_NOT_CROWD = 'not_crowd'

_CLASS_HAS_INSTANCES_LIST = dataset.CITYSCAPES_PANOPTIC_INFORMATION.class_has_instances_list
_PANOPTIC_LABEL_DIVISOR = dataset.CITYSCAPES_PANOPTIC_INFORMATION.panoptic_label_divisor
_FILENAME_PREFIX = 'dummy_000000_000000'


def create_test_data(annotation_path):
  """Creates cityscapes panoptic annotation, vps annotation and segment info.

  Our Video Panoptic Segmentation (VPS) encoding uses ID == semantic trainID *
  1000 + instance ID (starting at 1) with instance ID == 0 marking
  crowd regions.

  Args:
    annotation_path: The path to the annotation to be loaded.

  Returns:
    A tuple of cityscape annotation, vps annotation and segment infos.
  """
  # Convert panoptic labels to cityscapes label format.

  # Dictionary mapping converted panoptic annotation to its corresponding
  # Cityscapes label. Here the key is encoded by converting each RGB pixel
  # value to 1 * R + 256 * G + 256 * 256 * B.
  panoptic_label_to_cityscapes_label = {
      0: (_CITYSCAPES_IGNORE, _NOT_CROWD),
      31110: (_CITYSCAPES_CAR, _NOT_CROWD),
      31354: (_CITYSCAPES_CAR, _IS_CROWD),
      35173: (_CITYSCAPES_CAR, _NOT_CROWD),
      488314: (_CITYSCAPES_CAR, _IS_CROWD),
      549788: (_CITYSCAPES_CAR, _IS_CROWD),
      1079689: (_CITYSCAPES_CAR, _IS_CROWD),
      1341301: (_CITYSCAPES_CAR, _NOT_CROWD),
      1544590: (_CITYSCAPES_CAR, _NOT_CROWD),
      1926498: (_CITYSCAPES_CAR, _NOT_CROWD),
      4218944: (_CITYSCAPES_TREE, _NOT_CROWD),
      4251840: (_CITYSCAPES_SKY, _NOT_CROWD),
      6959003: (_CITYSCAPES_BUILDING, _NOT_CROWD),
      # To be merged with the building segment above.
      8396960: (_CITYSCAPES_BUILDING, _NOT_CROWD),
      8413312: (_CITYSCAPES_ROAD, _NOT_CROWD),
  }
  with tf.io.gfile.GFile(annotation_path, 'rb') as f:
    panoptic = data_utils.read_image(f.read())

  # Input panoptic annotation is RGB color coded, here we convert each pixel
  # to a unique number to avoid comparing 3-tuples.
  panoptic = np.dot(panoptic, [1, 256, 256 * 256])
  # Creates cityscapes panoptic map. Cityscapes use ID == semantic EvalId for
  # `stuff` segments and `thing` segments with `iscrowd` label, and
  # ID == semantic EvalId * 1000 + instance ID (starting from 0) for other
  # `thing` segments.
  cityscapes_panoptic = np.zeros_like(panoptic, dtype=np.int32)
  # Creates Video Panoptic Segmentation (VPS) map. We use ID == semantic
  # trainID * 1000 + instance ID (starting at 1) with instance ID == 0 marking
  # crowd regions.
  vps_panoptic = np.zeros_like(panoptic, dtype=np.int32)
  num_instances_per_class = collections.defaultdict(int)
  unique_labels = np.unique(panoptic)

  # Dictionary that maps segment id to segment info.
  segments_info = {}
  for label in unique_labels:
    cityscapes_label, is_crowd = panoptic_label_to_cityscapes_label[label]
    selected_pixels = panoptic == label

    if cityscapes_label == _CITYSCAPES_IGNORE:
      vps_panoptic[selected_pixels] = (
          _CITYSCAPES_IGNORE * _PANOPTIC_LABEL_DIVISOR)
      continue

    train_id, eval_id = tuple(cityscapes_label)
    cityscapes_id = eval_id
    vps_id = train_id * _PANOPTIC_LABEL_DIVISOR
    if train_id in _CLASS_HAS_INSTANCES_LIST:
      # `thing` class.
      if is_crowd != _IS_CROWD:
        cityscapes_id = (
            eval_id * _PANOPTIC_LABEL_DIVISOR +
            num_instances_per_class[train_id])
        # First instance should have ID 1.
        vps_id += num_instances_per_class[train_id] + 1
        num_instances_per_class[train_id] += 1

    cityscapes_panoptic[selected_pixels] = cityscapes_id
    vps_panoptic[selected_pixels] = vps_id
    pixel_area = int(np.sum(selected_pixels))
    if cityscapes_id in segments_info:
      logging.info('Merging segments with label %d into segment %d', label,
                   cityscapes_id)
      segments_info[cityscapes_id]['area'] += pixel_area
    else:
      segments_info[cityscapes_id] = {
          'area': pixel_area,
          'category_id': train_id,
          'id': cityscapes_id,
          'iscrowd': 1 if is_crowd == _IS_CROWD else 0,
      }

  cityscapes_panoptic = np.dstack([
      cityscapes_panoptic % 256, cityscapes_panoptic // 256,
      cityscapes_panoptic // 256 // 256
  ])
  vps_panoptic = np.dstack(
      [vps_panoptic % 256, vps_panoptic // 256, vps_panoptic // 256 // 256])
  return (cityscapes_panoptic.astype(np.uint8), vps_panoptic.astype(np.uint8),
          list(segments_info.values()))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_path = FLAGS.panoptic_annotation_path  # OSS: removed internal filename loading.
  panoptic_map, vps_map, segments_info = create_test_data(data_path)
  panoptic_map_filename = _FILENAME_PREFIX + '_gtFine_panoptic.png'
  panoptic_map_path = os.path.join(FLAGS.output_cityscapes_root, 'gtFine',
                                   'cityscapes_panoptic_dummy_trainId',
                                   panoptic_map_filename)

  gt_output_path = FLAGS.panoptic_gt_output_path  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(gt_output_path, 'wb') as f:
    Image.fromarray(vps_map).save(f, format='png')

  panoptic_map_path = panoptic_map_path  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(panoptic_map_path, 'wb') as f:
    Image.fromarray(panoptic_map).save(f, format='png')

  json_annotation = {
      'annotations': [{
          'file_name': _FILENAME_PREFIX + '_gtFine_panoptic.png',
          'image_id': _FILENAME_PREFIX,
          'segments_info': segments_info
      }]
  }
  json_annotation_path = os.path.join(FLAGS.output_cityscapes_root, 'gtFine',
                                      'cityscapes_panoptic_dummy_trainId.json')
  json_annotation_path = json_annotation_path  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(json_annotation_path, 'w') as f:
    json.dump(json_annotation, f, indent=2)


if __name__ == '__main__':
  app.run(main)
