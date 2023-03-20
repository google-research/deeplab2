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

"""Tests for build_cityscapes_data."""

import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from deeplab2.data import build_cityscapes_data


FLAGS = flags.FLAGS
_TEST_DATA_DIR = 'deeplab2/data/testdata'
_TEST_FILE_PREFIX = 'dummy_000000_000000'


class BuildCityscapesDataTest(tf.test.TestCase):

  def test_read_segments(self):
    cityscapes_root = os.path.join(_TEST_DATA_DIR)
    segments_dict = build_cityscapes_data._read_segments(
        cityscapes_root, dataset_split='dummy')
    self.assertIn(_TEST_FILE_PREFIX, segments_dict)
    _, segments = segments_dict[_TEST_FILE_PREFIX]
    self.assertLen(segments, 10)

  def test_generate_panoptic_label(self):
    FLAGS.treat_crowd_as_ignore = False  # Test a more complicated setting
    cityscapes_root = os.path.join(_TEST_DATA_DIR)
    segments_dict = build_cityscapes_data._read_segments(
        cityscapes_root, dataset_split='dummy')
    annotation_file_name, segments = segments_dict[_TEST_FILE_PREFIX]
    panoptic_annotation_file = build_cityscapes_data._get_panoptic_annotation(
        cityscapes_root, dataset_split='dummy',
        annotation_file_name=annotation_file_name)
    panoptic_label = build_cityscapes_data._generate_panoptic_label(
        panoptic_annotation_file, segments)

    # Check panoptic label matches golden file.
    golden_file_path = os.path.join(_TEST_DATA_DIR,
                                    'dummy_gt_for_vps.png')
    with tf.io.gfile.GFile(golden_file_path, 'rb') as f:
      golden_label = Image.open(f)
      # The PNG file is encoded by:
      #   color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
      golden_label = np.dot(np.asarray(golden_label), [1, 256, 256 * 256])

    np.testing.assert_array_equal(panoptic_label, golden_label)

if __name__ == '__main__':
  tf.test.main()
