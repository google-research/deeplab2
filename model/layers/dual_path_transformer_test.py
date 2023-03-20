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

"""Tests for transformer_layers."""

import functools
import os

import tensorflow as tf

from google.protobuf import text_format
# resources dependency
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import builder
from deeplab2.model.layers import dual_path_transformer


class TransformerLayersTest(tf.test.TestCase):

  def test_default_attention_operation_output_shape(self):
    layer = dual_path_transformer.AttentionOperation(
        'attention', 'relu', 'softmax')
    output = layer((tf.zeros([2, 8, 4225, 127]),
                    tf.zeros([2, 8, 422, 127]),
                    tf.zeros([2, 422, 8, 128])))
    self.assertListEqual(output.get_shape().as_list(), [2, 4225, 1024])

  def test_default_transformer_layer_output_shape(self):
    layer = dual_path_transformer.DualPathTransformerLayer()
    float_training_tensor = tf.constant(0.0, dtype=tf.float32)
    output = layer((tf.zeros([2, 25, 35, 126]),
                    tf.zeros([2, 127, 128]),
                    (),
                    float_training_tensor))
    self.assertListEqual(output[0].get_shape().as_list(), [2, 25, 35, 126])
    self.assertListEqual(output[1].get_shape().as_list(), [2, 127, 128])

  def test_kmeans_transformer_layer_output_shape(self):
    config_path = 'deeplab2/configs/example'
    def _read_proto_file(filename, proto):
      filename = filename  # OSS: removed internal filename loading.
      with tf.io.gfile.GFile(filename, 'r') as proto_file:
        return text_format.ParseLines(proto_file, proto)
    proto_filename = os.path.join(config_path,
                                  'example_coco_max_deeplab.textproto')
    config = _read_proto_file(proto_filename, config_pb2.ExperimentOptions())
    dataset_descriptor = dataset.MAP_NAME_TO_DATASET_INFO['coco_panoptic']
    auxiliary_predictor_func = functools.partial(
        builder.create_decoder,
        model_options=config.model_options,
        bn_layer=tf.keras.layers.BatchNormalization,
        ignore_label=dataset_descriptor.ignore_label,
        use_auxiliary_semantic_head=False)
    layer = dual_path_transformer.DualPathTransformerLayer(
        use_memory_self_attention=True,
        use_memory2pixel_feedback_attention=False,
        use_pixel2memory_feedback_attention=False,
        use_kmeans_cross_attention=True,
        auxiliary_predictor_func=auxiliary_predictor_func)
    float_training_tensor = tf.constant(0.0, dtype=tf.float32)
    output = layer((tf.zeros([2, 25, 35, 126]),
                    tf.zeros([2, 127, 128]),
                    (),
                    float_training_tensor))
    self.assertListEqual(output[0].get_shape().as_list(), [2, 25, 35, 126])
    self.assertListEqual(output[1].get_shape().as_list(), [2, 127, 128])

  def test_zero_feed_forward_network_output_shape(self):
    layer = dual_path_transformer.DualPathTransformerLayer(
        feed_forward_network_channels=0)
    float_training_tensor = tf.constant(0.0, dtype=tf.float32)
    output = layer((tf.zeros([2, 25, 35, 128]),
                    tf.zeros([2, 128, 128]),
                    (),
                    float_training_tensor))
    self.assertListEqual(output[0].get_shape().as_list(), [2, 25, 35, 128])
    self.assertListEqual(output[1].get_shape().as_list(), [2, 128, 128])

  def test_attention_types_output_shape(self):
    layer = dual_path_transformer.DualPathTransformerLayer(
        use_memory_self_attention=False,
        use_pixel2memory_feedback_attention=False)
    float_training_tensor = tf.constant(0.0, dtype=tf.float32)
    output = layer((tf.zeros([2, 25, 35, 128]),
                    tf.zeros([2, 128, 128]),
                    (),
                    float_training_tensor))
    self.assertListEqual(output[0].get_shape().as_list(), [2, 25, 35, 128])
    self.assertListEqual(output[1].get_shape().as_list(), [2, 128, 128])

if __name__ == '__main__':
  tf.test.main()
