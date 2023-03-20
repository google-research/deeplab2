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

"""Tests for build_step_data."""

import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from deeplab2.data import build_step_data

FLAGS = flags.FLAGS


class BuildStepDataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.data_dir = FLAGS.test_tmpdir
    self.height = 100
    self.width = 100
    self.sequence_id = '010'

  def _create_images(self, split):
    image_path = os.path.join(self.data_dir, build_step_data._IMAGE_FOLDER_NAME,
                              split, self.sequence_id)
    panoptic_map_path = os.path.join(self.data_dir,
                                     build_step_data._PANOPTIC_MAP_FOLDER_NAME,
                                     split, self.sequence_id)

    tf.io.gfile.makedirs(image_path)
    tf.io.gfile.makedirs(panoptic_map_path)
    self.panoptic_maps = {}
    for image_id in [101, 100]:
      self.panoptic_maps[image_id] = self._create_image_and_panoptic_map(
          image_path, panoptic_map_path, image_id)

  def _create_image_and_panoptic_map(self, image_path, panoptic_path, image_id):
    """Creates dummy images and panoptic maps."""
    # Dummy image.
    image = np.random.randint(
        0, 255, (self.height, self.width, 3), dtype=np.uint8)
    with tf.io.gfile.GFile(
        os.path.join(image_path, '%06d.png' % image_id), 'wb') as f:
      Image.fromarray(image).save(f, format='PNG')

    # Dummy panoptic map.
    semantic = np.random.randint(
        0, 20, (self.height, self.width), dtype=np.int32)
    instance = np.random.randint(
        0, 1000, (self.height, self.width), dtype=np.int32)
    encoded_panoptic_map = np.dstack(
        (semantic, instance // 256, instance % 256)).astype(np.uint8)
    with tf.io.gfile.GFile(
        os.path.join(panoptic_path, '%06d.png' % image_id), 'wb') as f:
      Image.fromarray(encoded_panoptic_map).save(f, format='PNG')
    decoded_panoptic_map = semantic * 1000 + instance
    return decoded_panoptic_map

  def test_build_step_dataset_correct(self):
    split = 'train'
    self._create_images(split)
    build_step_data._convert_dataset(
        step_root=self.data_dir,
        dataset_split=split,
        output_dir=FLAGS.test_tmpdir)
    # We will have 2 shards with each shard containing 1 image.
    num_shards = 2
    output_record = os.path.join(
        FLAGS.test_tmpdir, build_step_data._TF_RECORD_PATTERN %
        (split, 0, num_shards))
    self.assertTrue(tf.io.gfile.exists(output_record))

    # Parses tf record.
    image_ids = sorted(self.panoptic_maps)
    for i, raw_record in enumerate(
        tf.data.TFRecordDataset([output_record]).take(5)):
      image_id = image_ids[i]
      example = tf.train.Example.FromString(raw_record.numpy())
      panoptic_map = np.fromstring(
          example.features.feature['image/segmentation/class/encoded']
          .bytes_list.value[0],
          dtype=np.int32).reshape((self.height, self.width))
      np.testing.assert_array_equal(panoptic_map, self.panoptic_maps[image_id])
      self.assertEqual(
          example.features.feature['video/sequence_id'].bytes_list.value[0],
          b'010')
      self.assertEqual(
          example.features.feature['video/frame_id'].bytes_list.value[0],
          b'%06d' % image_id)

  def test_build_step_dataset_correct_with_two_frames(self):
    split = 'train'
    self._create_images(split)
    build_step_data._convert_dataset(
        step_root=self.data_dir,
        dataset_split=split,
        output_dir=FLAGS.test_tmpdir, use_two_frames=True)
    num_shards = 2
    output_record = os.path.join(
        FLAGS.test_tmpdir, build_step_data._TF_RECORD_PATTERN %
        (split, 0, num_shards))
    self.assertTrue(tf.io.gfile.exists(output_record))

    # Parses tf record.
    image_ids = sorted(self.panoptic_maps)
    for i, raw_record in enumerate(
        tf.data.TFRecordDataset([output_record]).take(5)):
      image_id = image_ids[i]
      example = tf.train.Example.FromString(raw_record.numpy())
      panoptic_map = np.fromstring(
          example.features.feature['image/segmentation/class/encoded']
          .bytes_list.value[0],
          dtype=np.int32).reshape((self.height, self.width))
      np.testing.assert_array_equal(panoptic_map, self.panoptic_maps[image_id])
      prev_panoptic_map = np.fromstring(
          example.features.feature['prev_image/segmentation/class/encoded']
          .bytes_list.value[0],
          dtype=np.int32).reshape((self.height, self.width))
      if i == 0:
        # First frame.
        np.testing.assert_array_equal(panoptic_map, prev_panoptic_map)
      else:
        # Not a first frame.
        np.testing.assert_array_equal(prev_panoptic_map, self.panoptic_maps[0])
      self.assertEqual(
          example.features.feature['video/sequence_id'].bytes_list.value[0],
          b'010')
      self.assertEqual(
          example.features.feature['video/frame_id'].bytes_list.value[0],
          b'%06d' % image_id)

  def test_build_step_dataset_with_two_frames_shared_by_sequence(self):
    split = 'val'
    self._create_images(split)
    build_step_data._convert_dataset(
        step_root=self.data_dir,
        dataset_split=split,
        output_dir=FLAGS.test_tmpdir, use_two_frames=True)
    # Only one shard since there is only one sequence for the val set.
    num_shards = 1
    output_record = os.path.join(
        FLAGS.test_tmpdir, build_step_data._TF_RECORD_PATTERN %
        (split, 0, num_shards))
    self.assertTrue(tf.io.gfile.exists(output_record))


if __name__ == '__main__':
  tf.test.main()
