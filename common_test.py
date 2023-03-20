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

"""Tests for common.py."""
import tensorflow as tf

from deeplab2 import common


class CommonTest(tf.test.TestCase):

  def test_constants_keys(self):
    self.assertEqual(common.PRED_PANOPTIC_KEY, 'panoptic_pred')
    self.assertEqual(common.PRED_SEMANTIC_KEY, 'semantic_pred')
    self.assertEqual(common.PRED_INSTANCE_CENTER_KEY, 'instance_center_pred')
    self.assertEqual(common.PRED_INSTANCE_KEY, 'instance_pred')

    self.assertEqual(common.PRED_SEMANTIC_LOGITS_KEY, 'semantic_logits')
    self.assertEqual(common.PRED_CENTER_HEATMAP_KEY, 'center_heatmap')
    self.assertEqual(common.PRED_OFFSET_MAP_KEY, 'offset_map')
    self.assertEqual(common.PRED_FRAME_OFFSET_MAP_KEY, 'frame_offset_map')

    self.assertEqual(common.GT_PANOPTIC_KEY, 'panoptic_gt')
    self.assertEqual(common.GT_SEMANTIC_KEY, 'semantic_gt')
    self.assertEqual(common.GT_INSTANCE_CENTER_KEY, 'instance_center_gt')
    self.assertEqual(common.GT_FRAME_OFFSET_KEY, 'frame_offset_gt')
    self.assertEqual(common.GT_INSTANCE_REGRESSION_KEY,
                     'instance_regression_gt')
    self.assertEqual(common.GT_PANOPTIC_RAW, 'panoptic_raw')
    self.assertEqual(common.GT_SEMANTIC_RAW, 'semantic_raw')
    self.assertEqual(common.GT_SIZE_RAW, 'size_raw')

    self.assertEqual(common.SEMANTIC_LOSS_WEIGHT_KEY, 'semantic_loss_weight')
    self.assertEqual(common.CENTER_LOSS_WEIGHT_KEY, 'center_loss_weight')
    self.assertEqual(common.REGRESSION_LOSS_WEIGHT_KEY,
                     'regression_loss_weight')
    self.assertEqual(common.FRAME_REGRESSION_LOSS_WEIGHT_KEY,
                     'frame_regression_loss_weight')

    self.assertEqual(common.RESIZED_IMAGE, 'resized_image')
    self.assertEqual(common.IMAGE, 'image')
    self.assertEqual(common.IMAGE_NAME, 'image_name')
    self.assertEqual(common.SEQUENCE_ID, 'sequence_id')
    self.assertEqual(common.FRAME_ID, 'frame_id')

    self.assertEqual(common.KEY_FRAME_ID, 'video/frame_id')
    self.assertEqual(common.KEY_SEQUENCE_ID, 'video/sequence_id')
    self.assertEqual(common.KEY_LABEL_FORMAT, 'image/segmentation/class/format')
    self.assertEqual(common.KEY_ENCODED_PREV_LABEL,
                     'prev_image/segmentation/class/encoded')
    self.assertEqual(common.KEY_ENCODED_LABEL,
                     'image/segmentation/class/encoded')
    self.assertEqual(common.KEY_IMAGE_CHANNELS, 'image/channels')
    self.assertEqual(common.KEY_IMAGE_WIDTH, 'image/width')
    self.assertEqual(common.KEY_IMAGE_HEIGHT, 'image/height')
    self.assertEqual(common.KEY_IMAGE_FORMAT, 'image/format')
    self.assertEqual(common.KEY_IMAGE_FILENAME, 'image/filename')
    self.assertEqual(common.KEY_ENCODED_PREV_IMAGE, 'prev_image/encoded')
    self.assertEqual(common.KEY_ENCODED_IMAGE, 'image/encoded')

  def test_multicamera_keys(self):
    test_camera_name = 'front'
    expected = {
        common.KEY_PER_CAMERA_ENCODED_IMAGE:
            'image/encoded/%s',
        common.KEY_PER_CAMERA_ENCODED_NEXT_IMAGE:
            'next_image/encoded/%s',
        common.KEY_PER_CAMERA_IMAGE_HEIGHT:
            'image/height/%s',
        common.KEY_PER_CAMERA_IMAGE_WIDTH:
            'image/width/%s',
        common.KEY_PER_CAMERA_ENCODED_LABEL:
            'image/segmentation/class/encoded/%s',
        common.KEY_PER_CAMERA_ENCODED_NEXT_LABEL:
            'next_image/segmentation/class/encoded/%s',
        common.KEY_PER_CAMERA_ENCODED_DEPTH:
            'image/depth/encoded/%s',
    }
    for key, val in expected.items():
      self.assertEqual(key % test_camera_name, val % test_camera_name)


if __name__ == '__main__':
  tf.test.main()
