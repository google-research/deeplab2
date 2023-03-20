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

"""Tests for build_coco_data."""

import json
import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from deeplab2.data import build_coco_data
from deeplab2.data import coco_constants

FLAGS = flags.FLAGS
_TEST_FILE_NAME = '000000123456.png'


class BuildCOCODataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.data_dir = FLAGS.test_tmpdir
    self.height = 100
    self.width = 100
    self.split = 'train'
    image_path = os.path.join(self.data_dir,
                              build_coco_data._FOLDERS_MAP[self.split]['image'])
    panoptic_map_path = os.path.join(self.data_dir,
                                     build_coco_data._FOLDERS_MAP
                                     [self.split]['label'])
    tf.io.gfile.makedirs(panoptic_map_path)
    panoptic_map_path = os.path.join(panoptic_map_path,
                                     'panoptic_%s2017' % self.split)

    tf.io.gfile.makedirs(image_path)
    tf.io.gfile.makedirs(panoptic_map_path)
    self.panoptic_maps = {}
    image_id = int(_TEST_FILE_NAME[:-4])
    self.panoptic_maps[image_id] = self._create_image_and_panoptic_map(
        image_path, panoptic_map_path, image_id)

  def _create_image_and_panoptic_map(self, image_path, panoptic_path, image_id):
    def id2rgb(id_map):
      id_map_copy = id_map.copy()
      rgb_shape = tuple(list(id_map.shape) + [3])
      rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
      for i in range(3):
        rgb_map[..., i] = id_map_copy % 256
        id_map_copy //= 256
      return rgb_map

    # Creates dummy images and panoptic maps.
    # Dummy image.
    image = np.random.randint(
        0, 255, (self.height, self.width, 3), dtype=np.uint8)
    with tf.io.gfile.GFile(
        os.path.join(image_path, '%012d.jpg' % image_id), 'wb') as f:
      Image.fromarray(image).save(f, format='JPEG')

    # Dummy panoptic map.
    semantic = np.random.randint(
        0, 201, (self.height, self.width), dtype=np.int32)
    instance_ = np.random.randint(
        0, 100, (self.height, self.width), dtype=np.int32)
    id_mapping = coco_constants.get_id_mapping()
    valid_semantic = id_mapping.keys()
    for i in range(201):
      if i not in valid_semantic:
        mask = (semantic == i)
        semantic[mask] = 0
        instance_[mask] = 0

    instance = instance_.copy()
    segments_info = []
    for sem in np.unique(semantic):
      ins_id = 1
      if sem == 0:
        continue
      if id_mapping[sem] in build_coco_data._CLASS_HAS_INSTANCE_LIST:
        for ins in np.unique(instance_[semantic == sem]):
          instance[np.logical_and(semantic == sem, instance_ == ins)] = ins_id
          area = np.logical_and(semantic == sem, instance_ == ins).sum()
          idx = sem * 256 + ins_id
          iscrowd = 0
          segments_info.append({
              'id': idx.tolist(),
              'category_id': sem.tolist(),
              'area': area.tolist(),
              'iscrowd': iscrowd,
          })
          ins_id += 1
      else:
        instance[semantic == sem] = 0
        area = (semantic == sem).sum()
        idx = sem * 256
        iscrowd = 0
        segments_info.append({
            'id': idx.tolist(),
            'category_id': sem.tolist(),
            'area': area.tolist(),
            'iscrowd': iscrowd,
        })

    encoded_panoptic_map = semantic * 256 + instance
    encoded_panoptic_map = id2rgb(encoded_panoptic_map)
    with tf.io.gfile.GFile(
        os.path.join(panoptic_path, '%012d.png' % image_id), 'wb') as f:
      Image.fromarray(encoded_panoptic_map).save(f, format='PNG')

    for i in range(201):
      if i in valid_semantic:
        mask = (semantic == i)
        semantic[mask] = id_mapping[i]

    decoded_panoptic_map = semantic * 256 + instance

    # Write json file
    json_annotation = {
        'annotations': [
            {
                'file_name': _TEST_FILE_NAME,
                'image_id': int(_TEST_FILE_NAME[:-4]),
                'segments_info': segments_info
            }
        ]
    }
    json_annotation_path = os.path.join(self.data_dir,
                                        build_coco_data._FOLDERS_MAP
                                        [self.split]['label'],
                                        'panoptic_%s2017.json' % self.split)
    with tf.io.gfile.GFile(json_annotation_path, 'w') as f:
      json.dump(json_annotation, f, indent=2)

    return decoded_panoptic_map

  def test_build_coco_dataset_correct(self):
    build_coco_data._convert_dataset(
        coco_root=self.data_dir,
        dataset_split=self.split,
        output_dir=FLAGS.test_tmpdir)
    output_record = os.path.join(
        FLAGS.test_tmpdir, '%s-%05d-of-%05d.tfrecord' %
        (self.split, 0, build_coco_data._NUM_SHARDS))
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

if __name__ == '__main__':
  tf.test.main()
