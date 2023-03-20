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

"""Tests for runner_utils.py."""

import os

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import deeplab
from deeplab2.trainer import runner_utils
# resources dependency

_CONFIG_PATH = 'deeplab2/configs/example'


def _read_proto_file(filename, proto):
  filename = filename  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(filename, 'r') as proto_file:
    return text_format.ParseLines(proto_file, proto)


def _create_model_from_test_proto(file_name,
                                  dataset_name='coco_panoptic'):
  proto_filename = os.path.join(_CONFIG_PATH, file_name)
  config = _read_proto_file(proto_filename, config_pb2.ExperimentOptions())
  return deeplab.DeepLab(config,
                         dataset.MAP_NAME_TO_DATASET_INFO[dataset_name]
                         ), config


class RunnerUtilsTest(tf.test.TestCase):

  def test_check_if_variable_in_backbone_with_max_deeplab(self):
    model, experiment_options = _create_model_from_test_proto(
        'example_coco_max_deeplab.textproto', dataset_name='coco_panoptic')
    train_crop_size = tuple(
        experiment_options.train_dataset_options.crop_size)
    input_tensor = tf.random.uniform(
        shape=(2, train_crop_size[0], train_crop_size[1], 3))
    _ = model(input_tensor, training=True)

    encoder = model.checkpoint_items['encoder']
    encoder_variable_names = [x.name for x in encoder.trainable_variables]
    encoder_name = experiment_options.model_options.backbone.name

    num_backbone_params = 0
    backbone_optimizer_inputs = []
    for variable in model.trainable_weights:
      if runner_utils.check_if_variable_in_backbone(variable, encoder_name,
                                                    encoder_variable_names):
        backbone_optimizer_inputs.append(variable)
        num_backbone_params += np.prod(variable.get_shape().as_list())
    # The number of Tensors in the backbone. We use this number in addition to
    # the number of parameters as a check of correctness.
    self.assertLen(backbone_optimizer_inputs, 301)
    # The same number of parameters as max_deeplab_s_backbone.
    self.assertEqual(num_backbone_params, 41343424)


if __name__ == '__main__':
  tf.test.main()
