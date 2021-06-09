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

r"""Converts Depth-aware Video Panoptic Segmentation (DVPS) data to sharded TFRecord file format with tf.train.Example protos.

The expected directory structure of the DVPS dataset should be as follows:

  + DVPS_ROOT
    + train | val
      - *_depth.png
      - *_gtFine_instanceTrainIds.png
      - *_leftImg8bit.png

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
     --panoptic_divisor=1000
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

flags.DEFINE_string('dvps_root', None, 'STEP dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')
flags.DEFINE_integer('panoptic_divisor', 1000,
                     'The divisor used to encode semantic and instance IDs.')

_PANOPTIC_DEPTH_FORMAT = 'raw'
_NUM_SHARDS = 10
_TF_RECORD_PATTERN = '%s-%05d-of-%05d.tfrecord'


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
  sequence_id = image_path.split('_')[0]
  image_id = image_path.split('_')[1]
  return sequence_id, image_id


def _get_images(dvps_root: str, dataset_split: str) -> Sequence[str]:
  """Gets files for the specified data type and dataset split.

  Args:
    dvps_root: String, Path to DVPS dataset root folder.
    dataset_split: String, dataset split ('train', 'val')

  Returns:
    A list of sorted file names under dvps_root and dataset_split.
  """
  search_files = os.path.join(dvps_root, dataset_split, '*_leftImg8bit.png',)
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
  return decoded_map.tobytes()


def _get_next_frame_path(image_path: str) -> Optional[str]:
  """Gets next frame path. If not exists, return None."""
  dir_name, image_name = os.path.split(image_path)
  image_name_split = image_name.split('_')[0:2]
  image_name_split[1] = '{:06d}'.format(int(image_name_split[1]) + 1)
  next_image_name = '_'.join(image_name_split) + '*'
  next_image_path = os.path.join(dir_name, next_image_name)
  next_image_path_search = tf.io.gfile.glob(next_image_path)
  # If the last frame, return None.
  if not next_image_path_search:
    return None
  next_image_name_search = [
      os.path.basename(path) for path in next_image_path_search]
  if 'leftImg8bit' in image_name:
    next_image_name = [
      name for name in next_image_name_search if name.endswith('bit.png')][0]
  else:
    next_image_name = [
      name for name in next_image_name_search if name.endswith('Ids.png')][0]
  next_image_path = os.path.join(dir_name, next_image_name)
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
  return data_utils.create_video_tfexample(
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


def _convert_dataset(dvps_root: str, dataset_split: str, output_dir: str,
                     panoptic_divisor: int):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dvps_root: String, Path to DVPS dataset root folder.
    dataset_split: String, the dataset split (e.g., train, val).
    output_dir: String, directory to write output TFRecords to.
    panoptic_divisor: Integer, the divisor to encode semantic and instance.
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
        sequence_id, image_id = _get_image_info_from_path(image_path)
        panoptic_map_path = image_path.replace('leftImg8bit',
                                               'gtFine_instanceTrainIds')
        depth_map_path = image_path.replace('leftImg8bit',
                                            'depth')
        example = _create_tfexample(image_path,
                                    panoptic_map_path,
                                    depth_map_path)
        if example is not None:
          tfrecord_writer.write(example.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.makedirs(FLAGS.output_dir)
  for dataset_split in ('train', 'val'):
    logging.info('Starts to processing DVPS dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.dvps_root, dataset_split, FLAGS.output_dir,
                     FLAGS.panoptic_divisor)


if __name__ == '__main__':
  app.run(main)
