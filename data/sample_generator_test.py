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

"""Tests for sample_generator."""

import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from deeplab2 import common
from deeplab2.data import data_utils
from deeplab2.data import dataset
from deeplab2.data import sample_generator

image_utils = tf.keras.preprocessing.image

flags.DEFINE_string(
    'panoptic_annotation_data',
    'deeplab2/data/testdata/',
    'Path to annotated test image.')
flags.DEFINE_bool('update_golden_data', False,
                  'Whether or not to update the golden data for testing.')

FLAGS = flags.FLAGS

_FILENAME_PREFIX = 'dummy_000000_000000'
_IMAGE_FOLDER = 'leftImg8bit/'
_TARGET_FOLDER = 'targets/'


def _get_groundtruth_image(computed_image_array, groundtruth_image_filename):
  if FLAGS.update_golden_data:
    image = Image.fromarray(tf.squeeze(computed_image_array).numpy())
    with tf.io.gfile.GFile(groundtruth_image_filename, mode='wb') as fp:
      image.save(fp)
    return computed_image_array

  with tf.io.gfile.GFile(groundtruth_image_filename, mode='rb') as fp:
    image = data_utils.read_image(fp.read())
    # If loaded image has 3 channels, the returned shape is [height, width, 3].
    # If loaded image has 1 channel, the returned shape is [height, width].
    image = np.squeeze(image_utils.img_to_array(image))
  return image


def _get_groundtruth_array(computed_image_array, groundtruth_image_filename):
  if FLAGS.update_golden_data:
    with tf.io.gfile.GFile(groundtruth_image_filename, mode='wb') as fp:
      np.save(fp, computed_image_array)
    return computed_image_array
  with tf.io.gfile.GFile(groundtruth_image_filename, mode='rb') as fp:
    # If loaded data has C>1 channels, the returned shape is [height, width, C].
    # If loaded data has 1 channel, the returned shape is [height, width].
    array = np.squeeze(np.load(fp))
  return array


class PanopticSampleGeneratorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_img_data_dir = os.path.join(
        FLAGS.test_srcdir,
        FLAGS.panoptic_annotation_data,
        _IMAGE_FOLDER)
    self._test_gt_data_dir = os.path.join(
        FLAGS.test_srcdir,
        FLAGS.panoptic_annotation_data)
    self._test_target_data_dir = os.path.join(
        FLAGS.test_srcdir,
        FLAGS.panoptic_annotation_data,
        _TARGET_FOLDER)
    image_path = self._test_img_data_dir + _FILENAME_PREFIX + '_leftImg8bit.png'
    with tf.io.gfile.GFile(image_path, 'rb') as image_file:
      rgb_image = data_utils.read_image(image_file.read())
    self._rgb_image = tf.convert_to_tensor(np.array(rgb_image))
    label_path = self._test_gt_data_dir + 'dummy_gt_for_vps.png'
    with tf.io.gfile.GFile(label_path, 'rb') as label_file:
      label = data_utils.read_image(label_file.read())
    self._label = tf.expand_dims(tf.convert_to_tensor(
        np.dot(np.array(label), [1, 256, 256 * 256])), -1)

  def test_input_generator(self):
    tf.random.set_seed(0)
    np.random.seed(0)
    small_instances = {'threshold': 4096, 'weight': 3.0}
    generator = sample_generator.PanopticSampleGenerator(
        dataset.CITYSCAPES_PANOPTIC_INFORMATION._asdict(),
        focus_small_instances=small_instances,
        is_training=True,
        crop_size=[769, 769],
        thing_id_mask_annotations=True)
    input_sample = {
        'image': self._rgb_image,
        'image_name': 'test_image',
        'label': self._label,
        'height': 800,
        'width': 800
    }
    sample = generator(input_sample)

    self.assertIn(common.IMAGE, sample)
    self.assertIn(common.GT_SEMANTIC_KEY, sample)
    self.assertIn(common.GT_PANOPTIC_KEY, sample)
    self.assertIn(common.GT_INSTANCE_CENTER_KEY, sample)
    self.assertIn(common.GT_INSTANCE_REGRESSION_KEY, sample)
    self.assertIn(common.GT_IS_CROWD, sample)
    self.assertIn(common.GT_THING_ID_MASK_KEY, sample)
    self.assertIn(common.GT_THING_ID_CLASS_KEY, sample)
    self.assertIn(common.SEMANTIC_LOSS_WEIGHT_KEY, sample)
    self.assertIn(common.CENTER_LOSS_WEIGHT_KEY, sample)
    self.assertIn(common.REGRESSION_LOSS_WEIGHT_KEY, sample)

    self.assertListEqual(sample[common.IMAGE].shape.as_list(), [769, 769, 3])
    self.assertListEqual(sample[common.GT_SEMANTIC_KEY].shape.as_list(),
                         [769, 769])
    self.assertListEqual(sample[common.GT_PANOPTIC_KEY].shape.as_list(),
                         [769, 769])
    self.assertListEqual(sample[common.GT_INSTANCE_CENTER_KEY].shape.as_list(),
                         [769, 769])
    self.assertListEqual(
        sample[common.GT_INSTANCE_REGRESSION_KEY].shape.as_list(),
        [769, 769, 2])
    self.assertListEqual(sample[common.GT_IS_CROWD].shape.as_list(), [769, 769])
    self.assertListEqual(sample[common.GT_THING_ID_MASK_KEY].shape.as_list(),
                         [769, 769])
    self.assertListEqual(sample[common.GT_THING_ID_CLASS_KEY].shape.as_list(),
                         [128])
    self.assertListEqual(
        sample[common.SEMANTIC_LOSS_WEIGHT_KEY].shape.as_list(), [769, 769])
    self.assertListEqual(sample[common.CENTER_LOSS_WEIGHT_KEY].shape.as_list(),
                         [769, 769])
    self.assertListEqual(
        sample[common.REGRESSION_LOSS_WEIGHT_KEY].shape.as_list(),
        [769, 769])

    gt_sem = sample[common.GT_SEMANTIC_KEY]
    gt_pan = sample[common.GT_PANOPTIC_KEY]
    gt_center = tf.cast(sample[common.GT_INSTANCE_CENTER_KEY] * 255, tf.uint8)
    gt_is_crowd = sample[common.GT_IS_CROWD]
    gt_thing_id_mask = sample[common.GT_THING_ID_MASK_KEY]
    gt_thing_id_class = sample[common.GT_THING_ID_CLASS_KEY]
    image = tf.cast(sample[common.IMAGE], tf.uint8)

    # semantic weights can be in range of [0, 3] in this example.
    semantic_weights = tf.cast(sample[common.SEMANTIC_LOSS_WEIGHT_KEY] * 85,
                               tf.uint8)
    center_weights = tf.cast(sample[common.CENTER_LOSS_WEIGHT_KEY] * 255,
                             tf.uint8)
    offset_weights = tf.cast(sample[common.REGRESSION_LOSS_WEIGHT_KEY] * 255,
                             tf.uint8)

    np.testing.assert_almost_equal(
        image.numpy(),
        _get_groundtruth_image(
            image,
            self._test_target_data_dir + 'rgb_target.png'))
    np.testing.assert_almost_equal(
        gt_sem.numpy(),
        _get_groundtruth_image(
            gt_sem,
            self._test_target_data_dir + 'semantic_target.png'))
    # Save gt as png. Pillow is currently unable to correctly save the image as
    # 32bit, but uses 16bit which overflows.
    _ = _get_groundtruth_image(
        gt_pan, self._test_target_data_dir + 'panoptic_target.png')
    np.testing.assert_almost_equal(
        gt_pan.numpy(),
        _get_groundtruth_array(
            gt_pan,
            self._test_target_data_dir + 'panoptic_target.npy'))
    np.testing.assert_almost_equal(
        gt_thing_id_mask.numpy(),
        _get_groundtruth_array(
            gt_thing_id_mask,
            self._test_target_data_dir + 'thing_id_mask_target.npy'))
    np.testing.assert_almost_equal(
        gt_thing_id_class.numpy(),
        _get_groundtruth_array(
            gt_thing_id_class,
            self._test_target_data_dir + 'thing_id_class_target.npy'))
    np.testing.assert_almost_equal(
        gt_center.numpy(),
        _get_groundtruth_image(
            gt_center,
            self._test_target_data_dir + 'center_target.png'))
    np.testing.assert_almost_equal(
        sample[common.GT_INSTANCE_REGRESSION_KEY].numpy(),
        _get_groundtruth_array(
            sample[common.GT_INSTANCE_REGRESSION_KEY].numpy(),
            self._test_target_data_dir + 'offset_target.npy'))
    np.testing.assert_array_equal(
        gt_is_crowd.numpy(),
        _get_groundtruth_array(gt_is_crowd.numpy(),
                               self._test_target_data_dir + 'is_crowd.npy'))
    np.testing.assert_almost_equal(
        semantic_weights.numpy(),
        _get_groundtruth_image(
            semantic_weights,
            self._test_target_data_dir + 'semantic_weights.png'))
    np.testing.assert_almost_equal(
        center_weights.numpy(),
        _get_groundtruth_image(
            center_weights,
            self._test_target_data_dir + 'center_weights.png'))
    np.testing.assert_almost_equal(
        offset_weights.numpy(),
        _get_groundtruth_image(
            offset_weights,
            self._test_target_data_dir + 'offset_weights.png'))

  def test_input_generator_eval(self):
    tf.random.set_seed(0)
    np.random.seed(0)
    small_instances = {'threshold': 4096, 'weight': 3.0}
    generator = sample_generator.PanopticSampleGenerator(
        dataset.CITYSCAPES_PANOPTIC_INFORMATION._asdict(),
        focus_small_instances=small_instances,
        is_training=False,
        crop_size=[800, 800])
    input_sample = {
        'image': self._rgb_image,
        'image_name': 'test_image',
        'label': self._label,
        'height': 800,
        'width': 800
    }
    sample = generator(input_sample)

    self.assertIn(common.GT_SEMANTIC_RAW, sample)
    self.assertIn(common.GT_PANOPTIC_RAW, sample)
    self.assertIn(common.GT_IS_CROWD_RAW, sample)

    gt_sem_raw = sample[common.GT_SEMANTIC_RAW]
    gt_pan_raw = sample[common.GT_PANOPTIC_RAW]
    gt_is_crowd_raw = sample[common.GT_IS_CROWD_RAW]

    self.assertListEqual(gt_sem_raw.shape.as_list(), [800, 800])
    self.assertListEqual(gt_pan_raw.shape.as_list(), [800, 800])
    self.assertListEqual(gt_is_crowd_raw.shape.as_list(), [800, 800])

    np.testing.assert_almost_equal(
        gt_sem_raw.numpy(),
        _get_groundtruth_image(
            gt_sem_raw,
            self._test_target_data_dir + 'eval_semantic_target.png'))
    np.testing.assert_almost_equal(
        gt_pan_raw.numpy(),
        _get_groundtruth_array(
            gt_pan_raw,
            self._test_target_data_dir + 'eval_panoptic_target.npy'))
    np.testing.assert_almost_equal(
        gt_is_crowd_raw.numpy(),
        _get_groundtruth_array(gt_is_crowd_raw, self._test_target_data_dir +
                               'eval_is_crowd.npy'))


if __name__ == '__main__':
  tf.test.main()
