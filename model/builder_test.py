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

"""Tests for model.builder."""

import os
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.model import builder
from deeplab2.model.decoder import motion_deeplab_decoder
from deeplab2.model.encoder import axial_resnet_instances
from deeplab2.model.encoder import mobilenet
# resources dependency


_CONFIG_PATH = 'deeplab2/configs/example'


def _read_proto_file(filename, proto):
  filename = filename  # OSS: removed internal filename loading.
  with tf.io.gfile.GFile(filename, 'r') as proto_file:
    return text_format.ParseLines(proto_file, proto)


class BuilderTest(tf.test.TestCase, parameterized.TestCase):

  def test_resnet50_encoder_creation(self):
    backbone_options = config_pb2.ModelOptions.BackboneOptions(
        name='resnet50', output_stride=32)
    encoder = builder.create_encoder(
        backbone_options,
        tf.keras.layers.experimental.SyncBatchNormalization)
    self.assertIsInstance(encoder, axial_resnet_instances.ResNet50)

  @parameterized.parameters('mobilenet_v3_large', 'mobilenet_v3_small')
  def test_mobilenet_encoder_creation(self, model_name):
    backbone_options = config_pb2.ModelOptions.BackboneOptions(
        name=model_name, use_squeeze_and_excite=True, output_stride=32)
    encoder = builder.create_encoder(
        backbone_options,
        tf.keras.layers.experimental.SyncBatchNormalization)
    self.assertIsInstance(encoder, mobilenet.MobileNet)

  def test_resnet_encoder_creation(self):
    backbone_options = config_pb2.ModelOptions.BackboneOptions(
        name='max_deeplab_s', output_stride=32)
    encoder = builder.create_resnet_encoder(
        backbone_options,
        bn_layer=tf.keras.layers.experimental.SyncBatchNormalization)
    self.assertIsInstance(encoder, axial_resnet_instances.MaXDeepLabS)

  def test_decoder_creation(self):
    proto_filename = os.path.join(
        _CONFIG_PATH, 'example_kitti-step_motion_deeplab.textproto')
    model_options = _read_proto_file(proto_filename, config_pb2.ModelOptions())
    motion_decoder = builder.create_decoder(
        model_options, tf.keras.layers.experimental.SyncBatchNormalization,
        ignore_label=255)
    self.assertIsInstance(motion_decoder,
                          motion_deeplab_decoder.MotionDeepLabDecoder)


if __name__ == '__main__':
  tf.test.main()
