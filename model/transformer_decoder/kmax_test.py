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

"""Tests for kMaX transformer decoder."""

import functools
import os

import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import builder
from deeplab2.model.transformer_decoder import kmax
# resources dependency


class KMaXTransformerDecoderTest(tf.test.TestCase):

  def test_model_output_shape(self):
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

    model = kmax.KMaXTransformerDecoder(
        name='kmax_pixel_decoder',
        auxiliary_predictor_func=auxiliary_predictor_func)
    output = model({
        'stage1': tf.keras.Input(shape=(321, 321, 64)),
        'stage2': tf.keras.Input(shape=(161, 161, 128)),
        'stage3': tf.keras.Input(shape=(81, 81, 256)),
        'stage4': tf.keras.Input(shape=(41, 41, 512)),
        'stage5': tf.keras.Input(shape=(21, 21, 1024)),
        'decoder_stage1': tf.keras.Input(shape=(21, 21, 2048)),
        'decoder_stage2': tf.keras.Input(shape=(41, 41, 1024)),
        'decoder_stage3': tf.keras.Input(shape=(81, 81, 512)),
        'decoder_output': tf.keras.Input(shape=(161, 161, 256)),
    })

    self.assertListEqual(
        output['transformer_class_feature'].get_shape().as_list(),
        [None, 128, 256])
    self.assertListEqual(
        output['transformer_mask_feature'].get_shape().as_list(),
        [None, 128, 256])
    self.assertListEqual(output['feature_panoptic'].get_shape().as_list(),
                         [None, 161, 161, 256])
    self.assertListEqual(output['feature_semantic'].get_shape().as_list(),
                         [None, 21, 21, 1024])


if __name__ == '__main__':
  tf.test.main()
