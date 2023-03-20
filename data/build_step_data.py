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

r"""Converts STEP (KITTI-STEP or MOTChallenge-STEP) data to sharded TFRecord file format with tf.train.Example protos.

The expected directory structure of the STEP dataset should be as follows:

  + {KITTI | MOTChallenge}-STEP
    + images
       + train
         + sequence_id
           - *.{png|jpg}
           ...
       + val
       + test
    + panoptic_maps
       + train
         + sequence_id
           - *.png
           ...
       + val

The ground-truth panoptic map is encoded as the following in PNG format:

  R: semantic_id
  G: instance_id // 256
  B: instance % 256

See ./utils/create_step_panoptic_maps.py for more details of how we create the
panoptic map by merging semantic and instance maps.

The output Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded panoptic segmentation content.
  image/segmentation/class/format: segmentation encoding format.
  video/sequence_id: sequence ID of the frame.
  video/frame_id: ID of the frame of the video sequence.

The output panoptic segmentation map stored in the Example will be the raw bytes
of an int32 panoptic map, where each pixel is assigned to a panoptic ID:

  panoptic ID = semantic ID * label divisor (1000) + instance ID

where semantic ID will be the same with `category_id` (use TrainId) for
each segment, and ignore label for pixels not belong to any segment.

The instance ID will be 0 for pixels belonging to
  1) `stuff` class
  2) `thing` class with `iscrowd` label
  3) pixels with ignore label
and [1, label divisor) otherwise.

Example to run the scipt:

   python deeplab2/data/build_step_data.py \
     --step_root=${STEP_ROOT} \
     --output_dir=${OUTPUT_DIR}
"""

import math
import os

from typing import Iterator, Sequence, Tuple, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np

from PIL import Image

import tensorflow as tf

from deeplab2.data import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('step_root', None, 'STEP dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')
flags.DEFINE_bool(
    'use_two_frames', False, 'Flag to separate between 1 frame '
    'per TFExample or 2 consecutive frames per TFExample.')

_PANOPTIC_LABEL_FORMAT = 'raw'
_NUM_SHARDS = 10
_IMAGE_FOLDER_NAME = 'images'
_PANOPTIC_MAP_FOLDER_NAME = 'panoptic_maps'
_LABEL_MAP_FORMAT = 'png'
_INSTANCE_LABEL_DIVISOR = 1000
_ENCODED_INSTANCE_LABEL_DIVISOR = 256
_TF_RECORD_PATTERN = '%s-%05d-of-%05d.tfrecord'
_FRAME_ID_PATTERN = '%06d'


def _get_image_info_from_path(image_path: str) -> Tuple[str, str]:
  """Gets image info including sequence id and image id.

  Image path is in the format of '.../split/sequence_id/image_id.png',
  where `sequence_id` refers to the id of the video sequence, and `image_id` is
  the id of the image in the video sequence.

  Args:
    image_path: Absolute path of the image.

  Returns:
    sequence_id, and image_id as strings.
  """
  sequence_id = image_path.split('/')[-2]
  image_id = os.path.splitext(os.path.basename(image_path))[0]
  return sequence_id, image_id


def _get_images_per_shard(step_root: str, dataset_split: str,
                          sharded_by_sequence: bool) -> Iterator[Sequence[str]]:
  """Gets files for the specified data type and dataset split.

  Args:
    step_root: String, Path to STEP dataset root folder.
    dataset_split: String, dataset split ('train', 'val', 'test')
    sharded_by_sequence: Whether the images should be sharded by sequence or
      even split.

  Yields:
    A list of sorted file lists. Each inner list corresponds to one shard and is
    a list of files for this shard.
  """
  search_files = os.path.join(step_root, _IMAGE_FOLDER_NAME, dataset_split, '*',
                              '*')
  filenames = sorted(tf.io.gfile.glob(search_files))
  num_per_even_shard = int(math.ceil(len(filenames) / _NUM_SHARDS))

  sequence_ids = [os.path.basename(os.path.dirname(name)) for name in filenames]
  images_per_shard = []
  for i, name in enumerate(filenames):
    images_per_shard.append(name)
    shard_data = (i == len(filenames) - 1)
    # Sharded by sequence id.
    shard_data = shard_data or (sharded_by_sequence and
                                sequence_ids[i + 1] != sequence_ids[i])
    # Sharded evenly.
    shard_data = shard_data or (not sharded_by_sequence and
                                len(images_per_shard) == num_per_even_shard)
    if shard_data:
      yield images_per_shard
      images_per_shard = []


def _decode_panoptic_map(panoptic_map_path: str) -> Optional[str]:
  """Decodes the panoptic map from encoded image file.

  Args:
    panoptic_map_path: Path to the panoptic map image file.

  Returns:
    Panoptic map as an encoded int32 numpy array bytes or None if not existing.
  """
  if not tf.io.gfile.exists(panoptic_map_path):
    return None
  with tf.io.gfile.GFile(panoptic_map_path, 'rb') as f:
    panoptic_map = np.array(Image.open(f)).astype(np.int32)
  semantic_map = panoptic_map[:, :, 0]
  instance_map = (
      panoptic_map[:, :, 1] * _ENCODED_INSTANCE_LABEL_DIVISOR +
      panoptic_map[:, :, 2])
  panoptic_map = semantic_map * _INSTANCE_LABEL_DIVISOR + instance_map
  return panoptic_map.tobytes()


def _get_previous_frame_path(image_path: str) -> str:
  """Gets previous frame path. If not exists, duplicate it with image_path."""
  frame_id, frame_ext = os.path.splitext(os.path.basename(image_path))
  folder_dir = os.path.dirname(image_path)
  prev_frame_id = _FRAME_ID_PATTERN % (int(frame_id) - 1)
  prev_image_path = os.path.join(folder_dir, prev_frame_id + frame_ext)
  # If first frame, duplicates it.
  if not tf.io.gfile.exists(prev_image_path):
    tf.compat.v1.logging.warn(
        'Could not find previous frame %s of frame %d, duplicate the previous '
        'frame with the current frame.', prev_image_path, int(frame_id))
    prev_image_path = image_path
  return prev_image_path


def _create_panoptic_tfexample(image_path: str,
                               panoptic_map_path: str,
                               use_two_frames: bool,
                               is_testing: bool = False) -> tf.train.Example:
  """Creates a TF example for each image.

  Args:
    image_path: Path to the image.
    panoptic_map_path: Path to the panoptic map (as an image file).
    use_two_frames: Whether to encode consecutive two frames in the Example.
    is_testing: Whether it is testing data. If so, skip adding label data.

  Returns:
    TF example proto.
  """
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    image_data = f.read()
  label_data = None
  if not is_testing:
    label_data = _decode_panoptic_map(panoptic_map_path)
  image_name = os.path.basename(image_path)
  image_format = image_name.split('.')[1].lower()
  sequence_id, frame_id = _get_image_info_from_path(image_path)
  prev_image_data = None
  prev_label_data = None
  if use_two_frames:
    # Previous image.
    prev_image_path = _get_previous_frame_path(image_path)
    with tf.io.gfile.GFile(prev_image_path, 'rb') as f:
      prev_image_data = f.read()
    # Previous panoptic map.
    if not is_testing:
      prev_panoptic_map_path = _get_previous_frame_path(panoptic_map_path)
      prev_label_data = _decode_panoptic_map(prev_panoptic_map_path)
  return data_utils.create_video_tfexample(
      image_data,
      image_format,
      image_name,
      label_format=_PANOPTIC_LABEL_FORMAT,
      sequence_id=sequence_id,
      image_id=frame_id,
      label_data=label_data,
      prev_image_data=prev_image_data,
      prev_label_data=prev_label_data)


def _convert_dataset(step_root: str,
                     dataset_split: str,
                     output_dir: str,
                     use_two_frames: bool = False):
  """Converts the specified dataset split to TFRecord format.

  Args:
    step_root: String, Path to STEP dataset root folder.
    dataset_split: String, the dataset split (e.g., train, val).
    output_dir: String, directory to write output TFRecords to.
    use_two_frames: Whether to encode consecutive two frames in the Example.
  """
  # For val and test set, if we run with use_two_frames, we should create a
  # sorted tfrecord per sequence.
  create_tfrecord_per_sequence = ('train'
                                  not in dataset_split) and use_two_frames
  is_testing = 'test' in dataset_split

  image_files_per_shard = list(
      _get_images_per_shard(step_root, dataset_split,
                            sharded_by_sequence=create_tfrecord_per_sequence))
  num_shards = len(image_files_per_shard)

  for shard_id, image_list in enumerate(image_files_per_shard):
    shard_filename = _TF_RECORD_PATTERN % (dataset_split, shard_id, num_shards)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      for image_path in image_list:
        sequence_id, image_id = _get_image_info_from_path(image_path)
        panoptic_map_path = os.path.join(
            step_root, _PANOPTIC_MAP_FOLDER_NAME, dataset_split, sequence_id,
            '%s.%s' % (image_id, _LABEL_MAP_FORMAT))
        example = _create_panoptic_tfexample(image_path, panoptic_map_path,
                                             use_two_frames, is_testing)
        tfrecord_writer.write(example.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.makedirs(FLAGS.output_dir)
  for dataset_split in ('train', 'val', 'test'):
    logging.info('Starts to processing STEP dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.step_root, dataset_split, FLAGS.output_dir,
                     FLAGS.use_two_frames)


if __name__ == '__main__':
  app.run(main)
