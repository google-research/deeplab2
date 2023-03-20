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

"""Tests for kmax_deeplab."""

import os

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import kmax_deeplab
from deeplab2.model import utils
# resources dependency

_CONFIG_PATH = 'deeplab2/configs/example'


def _read_proto_file(filename, proto):
  filename = filename  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(filename, 'r') as proto_file:
    return text_format.ParseLines(proto_file, proto)


def _create_model_from_test_proto(file_name,
                                  dataset_name='cityscapes_panoptic'):
  proto_filename = os.path.join(_CONFIG_PATH, file_name)
  config = _read_proto_file(proto_filename, config_pb2.ExperimentOptions())
  return kmax_deeplab.KMaXDeepLab(
      config,
      dataset.MAP_NAME_TO_DATASET_INFO[dataset_name]), config


class DeeplabTest(tf.test.TestCase):

  def test_deeplab_with_kmax_convnext_base(self):
    model, experiment_options = _create_model_from_test_proto(
        'example_coco_kmax_meta_convnext.textproto',
        dataset_name='coco_panoptic')
    train_crop_size = tuple(experiment_options.train_dataset_options.crop_size)
    input_tensor = tf.random.uniform(
        shape=(2, train_crop_size[0], train_crop_size[1], 3))
    stride_4_size = utils.scale_mutable_sequence(train_crop_size, 0.25)
    expected_semantic_shape = [
        2, stride_4_size[0], stride_4_size[1], experiment_options.model_options.
        max_deeplab.auxiliary_semantic_head.output_channels]
    expected_transformer_class_logits_shape = [
        2, 128, experiment_options.model_options.
        max_deeplab.auxiliary_semantic_head.output_channels]
    expected_pixel_space_normalized_feature_shape = [
        2, stride_4_size[0], stride_4_size[1], experiment_options.model_options.
        max_deeplab.pixel_space_head.output_channels]
    expected_pixel_space_mask_logits_shape = [
        2, stride_4_size[0], stride_4_size[1], 128]
    resulting_dict = model(input_tensor, training=True)
    self.assertListEqual(
        resulting_dict[common.PRED_SEMANTIC_LOGITS_KEY].shape.as_list(),
        expected_semantic_shape)
    self.assertListEqual(
        resulting_dict[
            common.PRED_TRANSFORMER_CLASS_LOGITS_KEY].shape.as_list(),
        expected_transformer_class_logits_shape)
    self.assertListEqual(
        resulting_dict[
            common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY].shape.as_list(),
        expected_pixel_space_normalized_feature_shape)
    self.assertListEqual(
        resulting_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY].shape.as_list(),
        expected_pixel_space_mask_logits_shape)
    num_params = 0
    for v in model.trainable_weights:
      params = np.prod(v.get_shape().as_list())
      # Exclude the auxiliary semantic head.
      if 'auxiliary_semantic' not in v.name:
        num_params += params
    self.assertEqual(num_params, 121513304)


if __name__ == '__main__':
  tf.test.main()
