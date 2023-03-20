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

r"""Converts Depth-aware Video Panoptic Segmentation (DVPS) data to sharded TFRecord file format with tf.train.Example protos.

The expected directory structure of the DVPS dataset should be as follows:

  + DVPS_ROOT
    + train | val
      - ground-truth depth maps (*_depth.png)
      - ground-truth panoptic maps (*_gtFine_instanceTrainIds.png)
      - images (*_leftImg8bit.png)
    + test
      - images (*_leftImg8bit.png)

The ground-truth panoptic map is encoded as the following in PNG format:

  panoptic ID = semantic ID * panoptic divisor (1000) + instance ID


The output Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded panoptic segmentation content.
  image/segmentation/class/format: segmentation encoding format.
  image/depth/encoded: encoded depth content.
  image/depth/format: depth encoding format.
  video/sequence_id: sequence ID of the frame.
  video/frame_id: ID of the frame of the video sequence.
  next_image/encoded: encoded next-frame image content.
  next_image/segmentation/class/encoded: encoded panoptic segmentation content
    of the next frame.

The output panoptic segmentation map stored in the Example will be the raw bytes
of an int32 panoptic map, where each pixel is assigned to a panoptic ID:

  panoptic ID = semantic ID * panoptic divisor (1000) + instance ID

where semantic ID will be the same with `category_id` for each segment, and
ignore label for pixels not belong to any segment.

The depth map will be the raw bytes of an int32 depth map, where each pixel is:

  depth map = depth ground truth * 256

Example to run the scipt:

   python deeplab2/data/build_dvps_data.py \
     --dvps_root=${DVPS_ROOT} \
     --output_dir=${OUTPUT_DIR} \
     --panoptic_divisor=${PANOPTIC_DIVISOR}
"""

import math
import os

from typing import Sequence, Tuple, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np

from PIL import Image

import tensorflow as tf

from deeplab2.data import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dvps_root', None, 'DVPS dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')

flags.DEFINE_integer('panoptic_divisor', default=65536,
                     help='Panoptic divisor used to encode 3-channel label.')

_ENCODED_INSTANCE_LABEL_DIVISOR = 256
_PANOPTIC_DEPTH_FORMAT = 'raw'
_NUM_SHARDS = 1000
_TF_RECORD_PATTERN = '%s-%05d-of-%05d.tfrecord'
_IMAGE_SUFFIX = '_leftImg8bit.png'
_LABEL_SUFFIX = '_gtFine_instanceTrainIds.png'
_DEPTH_SUFFIX = '_depth.png'


def _get_image_info_from_path(image_path: str) -> Tuple[str, str]:
  """Gets image info including sequence id and image id.

  Image path is in the format of '{sequence_id}_{image_id}_*.png',
  where `sequence_id` refers to the id of the video sequence, and `image_id` is
  the id of the image in the video sequence.

  Args:
    image_path: Absolute path of the image.

  Returns:
    sequence_id, and image_id as strings.
  """
  image_path = os.path.basename(image_path)
  return tuple(image_path.split('_')[:2])


def _get_images(dvps_root: str, dataset_split: str) -> Sequence[str]:
  """Gets files for the specified data type and dataset split.

  Args:
    dvps_root: String, path to DVPS dataset root folder.
    dataset_split: String, dataset split ('train', 'val', 'test').

  Returns:
    A list of sorted file names under dvps_root and dataset_split.
  """
  search_files = os.path.join(dvps_root, dataset_split, '*' + _IMAGE_SUFFIX)
  filenames = tf.io.gfile.glob(search_files)
  return sorted(filenames)


def _decode_panoptic_or_depth_map(map_path: str) -> Optional[str]:
  """Decodes the panoptic or depth map from encoded image file.

  Args:
    map_path: Path to the panoptic or depth map image file.

  Returns:
    Panoptic or depth map as an encoded int32 numpy array bytes or None if not
      existing.
  """
  if not tf.io.gfile.exists(map_path):
    return None
  with tf.io.gfile.GFile(map_path, 'rb') as f:
    decoded_map = np.array(Image.open(f)).astype(np.int32)
  if FLAGS.panoptic_divisor > 0 and map_path.endswith(_LABEL_SUFFIX):
    semantic_map = decoded_map[:, :, 0]
    instance_map = (
        decoded_map[:, :, 1] * _ENCODED_INSTANCE_LABEL_DIVISOR +
        decoded_map[:, :, 2])
    decoded_map = semantic_map * FLAGS.panoptic_divisor + instance_map
  return decoded_map.tobytes()


def _get_next_frame_path(image_path: str) -> Optional[str]:
  """Gets next frame path.

  If not exists, return None.

  The files are named {sequence_id}_{frame_id}*. To get the path of the next
  frame, this function keeps sequence_id and increase the frame_id by 1. It
  finds all the files matching this pattern, and returns the corresponding
  file path matching the input type.

  Args:
    image_path: String, path to the image.

  Returns:
    A string for the path of the next frame of the given image path or None if
      the given image path is the last frame of the sequence.
  """
  sequence_id, image_id = _get_image_info_from_path(image_path)
  next_image_id = '{:06d}'.format(int(image_id) + 1)
  next_image_name = sequence_id + '_' + next_image_id
  next_image_path = None
  for suffix in (_IMAGE_SUFFIX, _LABEL_SUFFIX):
    if image_path.endswith(suffix):
      next_image_path = os.path.join(
          os.path.dirname(image_path), next_image_name + suffix)
      if not tf.io.gfile.exists(next_image_path):
        return None
  return next_image_path


def _create_tfexample(image_path: str, panoptic_map_path: str,
                      depth_map_path: str) -> Optional[tf.train.Example]:
  """Creates a TF example for each image.

  Args:
    image_path: Path to the image.
    panoptic_map_path: Path to the panoptic map (as an image file).
    depth_map_path: Path to the depth map (as an image file).

  Returns:
    TF example proto.
  """
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    image_data = f.read()
  label_data = _decode_panoptic_or_depth_map(panoptic_map_path)
  depth_data = _decode_panoptic_or_depth_map(depth_map_path)
  image_name = os.path.basename(image_path)
  image_format = image_name.split('.')[1].lower()
  sequence_id, frame_id = _get_image_info_from_path(image_path)
  next_image_data = None
  next_label_data = None
  # Next image.
  next_image_path = _get_next_frame_path(image_path)
  # If there is no next image, no examples will be created.
  if next_image_path is None:
    return None
  with tf.io.gfile.GFile(next_image_path, 'rb') as f:
    next_image_data = f.read()
  # Next panoptic map.
  next_panoptic_map_path = _get_next_frame_path(panoptic_map_path)
  next_label_data = _decode_panoptic_or_depth_map(next_panoptic_map_path)
  return data_utils.create_video_and_depth_tfexample(
      image_data,
      image_format,
      image_name,
      label_format=_PANOPTIC_DEPTH_FORMAT,
      sequence_id=sequence_id,
      image_id=frame_id,
      label_data=label_data,
      next_image_data=next_image_data,
      next_label_data=next_label_data,
      depth_data=depth_data,
      depth_format=_PANOPTIC_DEPTH_FORMAT)


def _convert_dataset(dvps_root: str, dataset_split: str, output_dir: str):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dvps_root: String, path to DVPS dataset root folder.
    dataset_split: String, the dataset split (e.g., train, val, test).
    output_dir: String, directory to write output TFRecords to.
  """
  image_files = _get_images(dvps_root, dataset_split)
  num_images = len(image_files)

  num_per_shard = int(math.ceil(len(image_files) / _NUM_SHARDS))

  for shard_id in range(_NUM_SHARDS):
    shard_filename = _TF_RECORD_PATTERN % (dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        image_path = image_files[i]
        panoptic_map_path = image_path.replace(_IMAGE_SUFFIX, _LABEL_SUFFIX)
        depth_map_path = image_path.replace(_IMAGE_SUFFIX, _DEPTH_SUFFIX)
        example = _create_tfexample(image_path, panoptic_map_path,
                                    depth_map_path)
        if example is not None:
          tfrecord_writer.write(example.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.makedirs(FLAGS.output_dir)
  for dataset_split in ('train', 'val', 'test'):
    logging.info('Starts to processing DVPS dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.dvps_root, dataset_split, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
