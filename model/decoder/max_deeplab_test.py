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

"""Tests for max_deeplab."""

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.decoder import max_deeplab


def _create_max_deeplab_example_proto(num_non_void_classes=19):
  semantic_decoder = config_pb2.DecoderOptions(
      feature_key='feature_semantic', atrous_rates=[6, 12, 18])
  auxiliary_semantic_head = config_pb2.HeadOptions(
      output_channels=num_non_void_classes, head_channels=256)
  pixel_space_head = config_pb2.HeadOptions(
      output_channels=128, head_channels=256)
  max_deeplab_options = config_pb2.ModelOptions.MaXDeepLabOptions(
      pixel_space_head=pixel_space_head,
      auxiliary_semantic_head=auxiliary_semantic_head)
  # Add features from lowest to highest.
  max_deeplab_options.auxiliary_low_level.add(
      feature_key='res3', channels_project=64)
  max_deeplab_options.auxiliary_low_level.add(
      feature_key='res2', channels_project=32)
  return config_pb2.ModelOptions(
      decoder=semantic_decoder, max_deeplab=max_deeplab_options)


class MaXDeeplabTest(tf.test.TestCase):

  def test_max_deeplab_decoder_output_shape(self):
    num_non_void_classes = 19
    num_mask_slots = 127
    model_options = _create_max_deeplab_example_proto(
        num_non_void_classes=num_non_void_classes)
    decoder = max_deeplab.MaXDeepLab(
        max_deeplab_options=model_options.max_deeplab,
        ignore_label=255,
        decoder_options=model_options.decoder)

    input_dict = {
        'res2':
            tf.random.uniform([2, 17, 17, 256]),
        'res3':
            tf.random.uniform([2, 9, 9, 512]),
        'transformer_class_feature':
            tf.random.uniform([2, num_mask_slots, 256]),
        'transformer_mask_feature':
            tf.random.uniform([2, num_mask_slots, 256]),
        'feature_panoptic':
            tf.random.uniform([2, 17, 17, 256]),
        'feature_semantic':
            tf.random.uniform([2, 5, 5, 2048])
    }
    resulting_dict = decoder(input_dict)
    self.assertListEqual(
        resulting_dict[common.PRED_SEMANTIC_LOGITS_KEY].shape.as_list(),
        [2, 17, 17, 19])  # Stride 4
    self.assertListEqual(
        resulting_dict[
            common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY].shape.as_list(),
        [2, 17, 17, 128])  # Stride 4
    self.assertListEqual(
        resulting_dict[
            common.PRED_TRANSFORMER_CLASS_LOGITS_KEY].shape.as_list(),
        # Non-void classes and a void class.
        [2, num_mask_slots, num_non_void_classes + 1])
    self.assertListEqual(
        resulting_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY].shape.as_list(),
        [2, 17, 17, num_mask_slots])  # Stride 4.


if __name__ == '__main__':
  tf.test.main()
