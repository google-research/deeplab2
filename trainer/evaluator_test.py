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

"""Tests for the evaluator."""

import os
import tempfile
from unittest import mock

from absl import flags
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2 import trainer_pb2
from deeplab2.data import data_utils
from deeplab2.data import dataset
from deeplab2.data import sample_generator
from deeplab2.model import deeplab
from deeplab2.model.loss import loss_builder
from deeplab2.trainer import evaluator
from deeplab2.trainer import runner_utils

# resources dependency

_CONFIG_PATH = 'deeplab2/configs/example'

flags.DEFINE_string(
    'panoptic_annotation_data',
    'deeplab2/data/testdata/',
    'Path to annotated test image.')

FLAGS = flags.FLAGS

_FILENAME_PREFIX = 'dummy_000000_000000'
_IMAGE_FOLDER = 'leftImg8bit/'


def _read_proto_file(filename, proto):
  filename = filename  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(filename, 'r') as proto_file:
    return text_format.ParseLines(proto_file, proto)


def _create_panoptic_deeplab_loss(dataset_info):
  semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
      name='softmax_cross_entropy')
  center_loss_options = trainer_pb2.LossOptions.SingleLossOptions(name='mse')
  regression_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
      name='l1')
  loss_options = trainer_pb2.LossOptions(
      semantic_loss=semantic_loss_options,
      center_loss=center_loss_options,
      regression_loss=regression_loss_options)

  loss_layer = loss_builder.DeepLabFamilyLoss(
      loss_options,
      deeplab_options=config_pb2.ModelOptions(),
      num_classes=dataset_info.num_classes,
      ignore_label=dataset_info.ignore_label,
      ignore_depth=dataset_info.ignore_depth,
      thing_class_ids=dataset_info.class_has_instances_list)
  return loss_layer


def _create_max_deeplab_loss(dataset_info):
  semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
      name='softmax_cross_entropy')
  pq_style_loss_options = trainer_pb2.LossOptions.SingleLossOptions()
  mask_id_cross_entropy_loss_options = (
      trainer_pb2.LossOptions.SingleLossOptions())
  instance_discrimination_loss_options = (
      trainer_pb2.LossOptions.SingleLossOptions())
  loss_options = trainer_pb2.LossOptions(
      semantic_loss=semantic_loss_options,
      pq_style_loss=pq_style_loss_options,
      mask_id_cross_entropy_loss=mask_id_cross_entropy_loss_options,
      instance_discrimination_loss=instance_discrimination_loss_options)
  loss_layer = loss_builder.DeepLabFamilyLoss(
      loss_options,
      deeplab_options=config_pb2.ModelOptions(),
      num_classes=dataset_info.num_classes,
      ignore_label=dataset_info.ignore_label,
      ignore_depth=dataset_info.ignore_depth,
      thing_class_ids=dataset_info.class_has_instances_list)
  return loss_layer


class RealDataEvaluatorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_img_data_dir = os.path.join(
        FLAGS.test_srcdir,
        FLAGS.panoptic_annotation_data,
        _IMAGE_FOLDER)
    self._test_gt_data_dir = os.path.join(
        FLAGS.test_srcdir,
        FLAGS.panoptic_annotation_data)
    image_path = self._test_img_data_dir + _FILENAME_PREFIX + '_leftImg8bit.png'
    with tf.io.gfile.GFile(image_path, 'rb') as image_file:
      rgb_image = data_utils.read_image(image_file.read())
    self._rgb_image = tf.convert_to_tensor(np.array(rgb_image))
    label_path = self._test_gt_data_dir + 'dummy_gt_for_vps.png'
    with tf.io.gfile.GFile(label_path, 'rb') as label_file:
      label = data_utils.read_image(label_file.read())
    self._label = tf.expand_dims(tf.convert_to_tensor(
        np.dot(np.array(label), [1, 256, 256 * 256])), -1)

  def test_evaluates_max_deeplab_model(self):
    tf.random.set_seed(0)
    np.random.seed(0)
    small_instances = {'threshold': 4096, 'weight': 1.0}
    generator = sample_generator.PanopticSampleGenerator(
        dataset.CITYSCAPES_PANOPTIC_INFORMATION._asdict(),
        focus_small_instances=small_instances,
        is_training=False,
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

    experiment_options_textproto = """
      experiment_name: "evaluation_test"
      eval_dataset_options {
        dataset: "cityscapes_panoptic"
        file_pattern: "EMPTY"
        batch_size: 1
        crop_size: 769
        crop_size: 769
        thing_id_mask_annotations: true
      }
      evaluator_options {
        continuous_eval_timeout: -1
        stuff_area_limit: 2048
        center_score_threshold: 0.1
        nms_kernel: 13
        save_predictions: true
        save_raw_predictions: false
      }
    """
    config = text_format.Parse(experiment_options_textproto,
                               config_pb2.ExperimentOptions())

    model_proto_filename = os.path.join(
        _CONFIG_PATH, 'example_coco_max_deeplab.textproto')
    model_config = _read_proto_file(model_proto_filename,
                                    config_pb2.ExperimentOptions())
    config.model_options.CopyFrom(model_config.model_options)
    config.model_options.max_deeplab.auxiliary_semantic_head.output_channels = (
        19)
    model = deeplab.DeepLab(config, dataset.CITYSCAPES_PANOPTIC_INFORMATION)
    pool_size = (49, 49)
    model.set_pool_size(pool_size)

    loss_layer = _create_max_deeplab_loss(
        dataset.CITYSCAPES_PANOPTIC_INFORMATION)
    global_step = tf.Variable(initial_value=0, dtype=tf.int64)

    batched_sample = {}
    for key, value in sample.items():
      batched_sample[key] = tf.expand_dims(value, axis=0)
    real_data = [batched_sample]

    with tempfile.TemporaryDirectory() as model_dir:
      with mock.patch.object(runner_utils, 'create_dataset'):
        ev = evaluator.Evaluator(
            config, model, loss_layer, global_step, model_dir)

        state = ev.eval_begin()
        # Verify that output directories are created.
        self.assertTrue(os.path.isdir(os.path.join(model_dir, 'vis')))

        step_outputs = ev.eval_step(iter(real_data))

        state = ev.eval_reduce(state, step_outputs)
        result = ev.eval_end(state)

    expected_metric_keys = {
        'losses/eval_' + common.TOTAL_LOSS,
        'losses/eval_' + common.SEMANTIC_LOSS,
        'losses/eval_' + common.PQ_STYLE_LOSS_CLASS_TERM,
        'losses/eval_' + common.PQ_STYLE_LOSS_MASK_DICE_TERM,
        'losses/eval_' + common.MASK_ID_CROSS_ENTROPY_LOSS,
        'losses/eval_' + common.INSTANCE_DISCRIMINATION_LOSS,
        'evaluation/iou/IoU',
        'evaluation/pq/PQ',
        'evaluation/pq/SQ',
        'evaluation/pq/RQ',
        'evaluation/pq/TP',
        'evaluation/pq/FN',
        'evaluation/pq/FP',
        'evaluation/ap/AP_Mask',
    }
    self.assertCountEqual(result.keys(), expected_metric_keys)
    self.assertSequenceEqual(result['losses/eval_total_loss'].shape, ())


class EvaluatorTest(tf.test.TestCase):

  def test_evaluates_panoptic_deeplab_model(self):
    experiment_options_textproto = """
      experiment_name: "evaluation_test"
      eval_dataset_options {
        dataset: "cityscapes_panoptic"
        file_pattern: "EMPTY"
        batch_size: 1
        crop_size: 1025
        crop_size: 2049
        # Skip resizing.
        min_resize_value: 0
        max_resize_value: 0
      }
      evaluator_options {
        continuous_eval_timeout: -1
        stuff_area_limit: 2048
        center_score_threshold: 0.1
        nms_kernel: 13
        save_predictions: true
        save_raw_predictions: false
      }
    """
    config = text_format.Parse(experiment_options_textproto,
                               config_pb2.ExperimentOptions())

    model_proto_filename = os.path.join(
        _CONFIG_PATH, 'example_cityscapes_panoptic_deeplab.textproto')
    model_config = _read_proto_file(model_proto_filename,
                                    config_pb2.ExperimentOptions())
    config.model_options.CopyFrom(model_config.model_options)
    model = deeplab.DeepLab(config, dataset.CITYSCAPES_PANOPTIC_INFORMATION)
    pool_size = (33, 65)
    model.set_pool_size(pool_size)

    loss_layer = _create_panoptic_deeplab_loss(
        dataset.CITYSCAPES_PANOPTIC_INFORMATION)
    global_step = tf.Variable(initial_value=0, dtype=tf.int64)

    fake_datum = {
        common.IMAGE:
            tf.zeros([1, 1025, 2049, 3]),
        common.RESIZED_IMAGE:
            tf.zeros([1, 1025, 2049, 3]),
        common.GT_SIZE_RAW:
            tf.constant([[1025, 2049]], dtype=tf.int32),
        common.GT_SEMANTIC_KEY:
            tf.zeros([1, 1025, 2049], dtype=tf.int32),
        common.GT_SEMANTIC_RAW:
            tf.zeros([1, 1025, 2049], dtype=tf.int32),
        common.GT_PANOPTIC_RAW:
            tf.zeros([1, 1025, 2049], dtype=tf.int32),
        common.GT_IS_CROWD_RAW:
            tf.zeros([1, 1025, 2049], dtype=tf.uint8),
        common.GT_INSTANCE_CENTER_KEY:
            tf.zeros([1, 1025, 2049], dtype=tf.float32),
        common.GT_INSTANCE_REGRESSION_KEY:
            tf.zeros([1, 1025, 2049, 2], dtype=tf.float32),
        common.IMAGE_NAME:
            'fake',
        common.SEMANTIC_LOSS_WEIGHT_KEY:
            tf.zeros([1, 1025, 2049], dtype=tf.float32),
        common.CENTER_LOSS_WEIGHT_KEY:
            tf.zeros([1, 1025, 2049], dtype=tf.float32),
        common.REGRESSION_LOSS_WEIGHT_KEY:
            tf.zeros([1, 1025, 2049], dtype=tf.float32),
    }
    fake_data = [fake_datum]

    with tempfile.TemporaryDirectory() as model_dir:
      with mock.patch.object(runner_utils, 'create_dataset'):
        ev = evaluator.Evaluator(
            config, model, loss_layer, global_step, model_dir)

        state = ev.eval_begin()
        # Verify that output directories are created.
        self.assertTrue(os.path.isdir(os.path.join(model_dir, 'vis')))

        step_outputs = ev.eval_step(iter(fake_data))

        state = ev.eval_reduce(state, step_outputs)
        result = ev.eval_end(state)

    expected_metric_keys = {
        'losses/eval_total_loss',
        'losses/eval_semantic_loss',
        'losses/eval_center_loss',
        'losses/eval_regression_loss',
        'evaluation/iou/IoU',
        'evaluation/pq/PQ',
        'evaluation/pq/SQ',
        'evaluation/pq/RQ',
        'evaluation/pq/TP',
        'evaluation/pq/FN',
        'evaluation/pq/FP',
        'evaluation/ap/AP_Mask',
    }
    self.assertCountEqual(result.keys(), expected_metric_keys)

    self.assertSequenceEqual(result['losses/eval_total_loss'].shape, ())
    self.assertEqual(result['losses/eval_total_loss'].numpy(), 0.0)


if __name__ == '__main__':
  tf.test.main()
