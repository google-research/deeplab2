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

"""Tests for the evaluator."""

import os
import tempfile
from unittest import mock

import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import deeplab
from deeplab2.trainer import evaluator
from deeplab2.trainer import runner_utils

# resources dependency

_CONFIG_PATH = 'deeplab2/configs/example'


def _read_proto_file(filename, proto):
  filename = filename  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(filename, 'r') as proto_file:
    return text_format.ParseLines(proto_file, proto)


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
        continuous_eval_timeout: 43200
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

    loss = mock.Mock(spec=tf.keras.losses.Loss)
    loss.return_value = tf.zeros([1])
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
    }
    fake_data = [fake_datum]

    with tempfile.TemporaryDirectory() as model_dir:
      with mock.patch.object(runner_utils, 'create_dataset'):
        ev = evaluator.Evaluator(config, model, loss, global_step, model_dir)

        state = ev.eval_begin()
        # Verify that output directories are created.
        self.assertTrue(os.path.isdir(os.path.join(model_dir, 'vis')))

        step_outputs = ev.eval_step(iter(fake_data))

        state = ev.eval_reduce(state, step_outputs)
        result = ev.eval_end(state)

    expected_metric_keys = {
        'losses/eval_loss',
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

    self.assertSequenceEqual(result['losses/eval_loss'].shape, ())
    self.assertEqual(result['losses/eval_loss'].numpy(), 0.0)


if __name__ == '__main__':
  tf.test.main()
