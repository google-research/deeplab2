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

"""Converts COCO data to sharded TFRecord file format with Example protos.

Please check
  ../g3doc/setup/coco.md
for instructions.
"""

import collections
import json
import math
import os

from typing import Sequence, Tuple, Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from deeplab2.data import coco_constants
from deeplab2.data import data_utils
from deeplab2.data import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('coco_root', None, 'coco dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')

flags.DEFINE_boolean('treat_crowd_as_ignore', True,
                     'Whether to apply ignore labels to crowd pixels in '
                     'panoptic label.')

_NUM_SHARDS = 1000


_SPLITS_TO_SIZES = dataset.COCO_PANOPTIC_INFORMATION.splits_to_sizes
_IGNORE_LABEL = dataset.COCO_PANOPTIC_INFORMATION.ignore_label
_CLASS_HAS_INSTANCE_LIST = dataset.COCO_PANOPTIC_INFORMATION.class_has_instances_list
_PANOPTIC_LABEL_DIVISOR = dataset.COCO_PANOPTIC_INFORMATION.panoptic_label_divisor
_CLASS_MAPPING = coco_constants.get_id_mapping()

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'train': {
        'image': 'train2017',
        'label': 'annotations',
    },
    'val': {
        'image': 'val2017',
        'label': 'annotations',
    },
    'test': {
        'image': 'test2017',
        'label': '',
    }
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'jpg',
    'label': 'png',
}
_PANOPTIC_LABEL_FORMAT = 'raw'


def _get_images(coco_root: str, dataset_split: str) -> Sequence[str]:
  """Gets files for the specified data type and dataset split.

  Args:
    coco_root: String, path to coco dataset root folder.
    dataset_split: String, dataset split ('train', 'val', 'test').

  Returns:
    A list of sorted file names.
  """
  pattern = '*.%s' % _DATA_FORMAT_MAP['image']
  search_files = os.path.join(
      coco_root, _FOLDERS_MAP[dataset_split]['image'], pattern)
  filenames = tf.io.gfile.glob(search_files)
  return sorted(filenames)


def _get_panoptic_annotation(coco_root: str, dataset_split: str,
                             annotation_file_name: str) -> str:
  panoptic_folder = 'panoptic_%s2017' % dataset_split
  return os.path.join(coco_root, _FOLDERS_MAP[dataset_split]['label'],
                      panoptic_folder, annotation_file_name)


def _read_segments(coco_root: str, dataset_split: str):
  """Reads segments information from json file.

  Args:
    coco_root: String, path to coco dataset root folder.
    dataset_split: String, dataset split.

  Returns:
    segments_dict: A dictionary that maps file prefix of annotation_file_name to
      a tuple of (panoptic annotation file name, segments). Please refer to
      _generate_panoptic_label() method on the detail structure of `segments`.

  Raises:
    ValueError: If found duplicated image id in annotations.
  """
  json_filename = os.path.join(
      coco_root, _FOLDERS_MAP[dataset_split]['label'],
      'panoptic_%s2017.json' % dataset_split)
  with tf.io.gfile.GFile(json_filename) as f:
    panoptic_dataset = json.load(f)

  segments_dict = {}
  for annotation in panoptic_dataset['annotations']:
    image_id = annotation['image_id']
    if image_id in segments_dict:
      raise ValueError('Image ID %s already exists' % image_id)
    annotation_file_name = annotation['file_name']
    segments = annotation['segments_info']

    segments_dict[os.path.splitext(annotation_file_name)[-2]] = (
        annotation_file_name, segments)

  return segments_dict


def _generate_panoptic_label(panoptic_annotation_file: str, segments:
                             Any) -> np.ndarray:
  """Creates panoptic label map from annotations.

  Args:
    panoptic_annotation_file: String, path to panoptic annotation.
    segments: A list of dictionaries containing information of every segment.
      Read from panoptic_${DATASET_SPLIT}2017.json. This method consumes
      the following fields in each dictionary:
        - id: panoptic id
        - category_id: semantic class id
        - area: pixel area of this segment
        - iscrowd: if this segment is crowd region

  Returns:
    A 2D numpy int32 array with the same height / width with panoptic
    annotation. Each pixel value represents its panoptic ID. Please refer to
    g3doc/setup/coco.md for more details about how panoptic ID is assigned.
  """
  with tf.io.gfile.GFile(panoptic_annotation_file, 'rb') as f:
    panoptic_label = data_utils.read_image(f.read())

  if panoptic_label.mode != 'RGB':
    raise ValueError('Expect RGB image for panoptic label, gets %s' %
                     panoptic_label.mode)

  panoptic_label = np.array(panoptic_label, dtype=np.int32)
  # COCO panoptic map is created by:
  #   color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
  panoptic_label = np.dot(panoptic_label, [1, 256, 256 * 256])

  semantic_label = np.ones_like(panoptic_label) * _IGNORE_LABEL
  instance_label = np.zeros_like(panoptic_label)
  # Running count of instances per semantic category.
  instance_count = collections.defaultdict(int)

  for segment in segments:
    selected_pixels = panoptic_label == segment['id']
    pixel_area = np.sum(selected_pixels)
    if pixel_area != segment['area']:
      raise ValueError('Expect %d pixels for segment %s, gets %d.' %
                       (segment['area'], segment, pixel_area))

    category_id = segment['category_id']

    # Map the category_id to contiguous ids
    category_id = _CLASS_MAPPING[category_id]

    semantic_label[selected_pixels] = category_id

    if category_id in _CLASS_HAS_INSTANCE_LIST:
      if segment['iscrowd']:
        # COCO crowd pixels will have instance ID of 0.
        if FLAGS.treat_crowd_as_ignore:
          semantic_label[selected_pixels] = _IGNORE_LABEL
        continue
      # Non-crowd pixels will have instance ID starting from 1.
      instance_count[category_id] += 1
      if instance_count[category_id] >= _PANOPTIC_LABEL_DIVISOR:
        raise ValueError('Too many instances for category %d in this image.' %
                         category_id)
      instance_label[selected_pixels] = instance_count[category_id]
    elif segment['iscrowd']:
      raise ValueError('Stuff class should not have `iscrowd` label.')

  panoptic_label = semantic_label * _PANOPTIC_LABEL_DIVISOR + instance_label
  return panoptic_label.astype(np.int32)


def _create_panoptic_label(coco_root: str, dataset_split: str, image_path: str,
                           segments_dict: Any
                           ) -> Tuple[str, str]:
  """Creates labels for panoptic segmentation.

  Args:
    coco_root: String, path to coco dataset root folder.
    dataset_split: String, dataset split ('train', 'val', 'test').
    image_path: String, path to the image file.
    segments_dict:
      Read from panoptic_${DATASET_SPLIT}2017.json. This method consumes
      the following fields in each dictionary:
        - id: panoptic id
        - category_id: semantic class id
        - area: pixel area of this segment
        - iscrowd: if this segment is crowd region

  Returns:
    A panoptic label where each pixel value represents its panoptic ID.
      Please refer to g3doc/setup/coco.md for more details about howpanoptic ID
      is assigned.
    A string indicating label format in TFRecord.
  """

  image_path = os.path.normpath(image_path)
  path_list = image_path.split(os.sep)
  file_name = path_list[-1]

  annotation_file_name, segments = segments_dict[
      os.path.splitext(file_name)[-2]]
  panoptic_annotation_file = _get_panoptic_annotation(coco_root,
                                                      dataset_split,
                                                      annotation_file_name)

  panoptic_label = _generate_panoptic_label(panoptic_annotation_file, segments)
  return panoptic_label.tostring(), _PANOPTIC_LABEL_FORMAT


def _convert_dataset(coco_root: str, dataset_split: str,
                     output_dir: str) -> None:
  """Converts the specified dataset split to TFRecord format.

  Args:
    coco_root: String, path to coco dataset root folder.
    dataset_split: String, the dataset split (one of `train`, `val` and `test`).
    output_dir: String, directory to write output TFRecords to.
  """
  image_files = _get_images(coco_root, dataset_split)

  num_images = len(image_files)

  if dataset_split != 'test':
    segments_dict = _read_segments(coco_root, dataset_split)

  num_per_shard = int(math.ceil(len(image_files) / _NUM_SHARDS))

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        # Read the image.
        with tf.io.gfile.GFile(image_files[i], 'rb') as f:
          image_data = f.read()

        if dataset_split == 'test':
          label_data, label_format = None, None
        else:
          label_data, label_format = _create_panoptic_label(
              coco_root, dataset_split, image_files[i], segments_dict)

        # Convert to tf example.
        image_path = os.path.normpath(image_files[i])
        path_list = image_path.split(os.sep)
        file_name = path_list[-1]
        file_prefix = os.path.splitext(file_name)[0]
        example = data_utils.create_tfexample(image_data,
                                              'jpeg',
                                              file_prefix, label_data,
                                              label_format)

        tfrecord_writer.write(example.SerializeToString())


def main(unused_argv: Sequence[str]) -> None:
  tf.io.gfile.makedirs(FLAGS.output_dir)

  for dataset_split in ('train', 'val', 'test'):
    logging.info('Starts processing dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.coco_root, dataset_split, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['coco_root', 'output_dir'])
  app.run(main)
