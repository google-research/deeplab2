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

"""Tests for panoptic_deeplab."""

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.decoder import panoptic_deeplab
from deeplab2.utils import test_utils


def _create_panoptic_deeplab_example_proto(num_classes=19):
  semantic_decoder = config_pb2.DecoderOptions(
      feature_key='res5', atrous_rates=[6, 12, 18])
  semantic_head = config_pb2.HeadOptions(
      output_channels=num_classes, head_channels=256)

  instance_decoder = config_pb2.DecoderOptions(
      feature_key='res5', decoder_channels=128, atrous_rates=[6, 12, 18])
  center_head = config_pb2.HeadOptions(
      output_channels=1, head_channels=32)
  regression_head = config_pb2.HeadOptions(
      output_channels=2, head_channels=32)

  instance_branch = config_pb2.InstanceOptions(
      instance_decoder_override=instance_decoder,
      center_head=center_head,
      regression_head=regression_head)

  panoptic_deeplab_options = config_pb2.ModelOptions.PanopticDeeplabOptions(
      semantic_head=semantic_head, instance=instance_branch)
  # Add features from lowest to highest.
  panoptic_deeplab_options.low_level.add(
      feature_key='res3', channels_project=64)
  panoptic_deeplab_options.low_level.add(
      feature_key='res2', channels_project=32)

  return config_pb2.ModelOptions(
      decoder=semantic_decoder, panoptic_deeplab=panoptic_deeplab_options)


def _create_expected_shape(input_shape, output_channels):
  output_shape = input_shape.copy()
  output_shape[3] = output_channels
  return output_shape


class PanopticDeeplabTest(tf.test.TestCase):

  def test_panoptic_deeplab_single_decoder_init_errors(self):
    with self.assertRaises(ValueError):
      _ = panoptic_deeplab.PanopticDeepLabSingleDecoder(
          high_level_feature_name='test',
          low_level_feature_names=['only_one_name'],  # Error: Only one name.
          low_level_channels_project=[64, 32],
          aspp_output_channels=256,
          decoder_output_channels=256,
          atrous_rates=[6, 12, 18],
          name='test_decoder')

    with self.assertRaises(ValueError):
      _ = panoptic_deeplab.PanopticDeepLabSingleDecoder(
          high_level_feature_name='test',
          low_level_feature_names=['one', 'two'],
          low_level_channels_project=[64],  # Error: Only one projection size.
          aspp_output_channels=256,
          decoder_output_channels=256,
          atrous_rates=[6, 12, 18],
          name='test_decoder')

  def test_panoptic_deeplab_single_decoder_call_errors(self):
    decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        high_level_feature_name='high',
        low_level_feature_names=['low_one', 'low_two'],
        low_level_channels_project=[64, 32],
        aspp_output_channels=256,
        decoder_output_channels=256,
        atrous_rates=[6, 12, 18],
        name='test_decoder')

    with self.assertRaises(KeyError):
      input_dict = {'not_high': tf.random.uniform(shape=(2, 32, 32, 512)),
                    'low_one': tf.random.uniform(shape=(2, 128, 128, 128)),
                    'low_two': tf.random.uniform(shape=(2, 256, 256, 64))}
      _ = decoder(input_dict)
    with self.assertRaises(KeyError):
      input_dict = {'high': tf.random.uniform(shape=(2, 32, 32, 512)),
                    'not_low_one': tf.random.uniform(shape=(2, 128, 128, 128)),
                    'low_two': tf.random.uniform(shape=(2, 256, 256, 64))}
      _ = decoder(input_dict)
    with self.assertRaises(KeyError):
      input_dict = {'high': tf.random.uniform(shape=(2, 32, 32, 512)),
                    'low_one': tf.random.uniform(shape=(2, 128, 128, 128)),
                    'not_low_two': tf.random.uniform(shape=(2, 256, 256, 64))}
      _ = decoder(input_dict)

  def test_panoptic_deeplab_single_decoder_reset_pooling(self):
    decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        high_level_feature_name='high',
        low_level_feature_names=['low_one', 'low_two'],
        low_level_channels_project=[64, 32],
        aspp_output_channels=256,
        decoder_output_channels=256,
        atrous_rates=[6, 12, 18],
        name='test_decoder')
    pool_size = (None, None)
    decoder.reset_pooling_layer()

    self.assertTupleEqual(decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  def test_panoptic_deeplab_single_decoder_set_pooling(self):
    decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        high_level_feature_name='high',
        low_level_feature_names=['low_one', 'low_two'],
        low_level_channels_project=[64, 32],
        aspp_output_channels=256,
        decoder_output_channels=256,
        atrous_rates=[6, 12, 18],
        name='test_decoder')

    pool_size = (10, 10)
    decoder.set_pool_size(pool_size)

    self.assertTupleEqual(decoder._aspp._aspp_pool._pool_size,
                          pool_size)

  def test_panoptic_deeplab_single_decoder_output_shape(self):
    decoder_channels = 256
    decoder = panoptic_deeplab.PanopticDeepLabSingleDecoder(
        high_level_feature_name='high',
        low_level_feature_names=['low_one', 'low_two'],
        low_level_channels_project=[64, 32],
        aspp_output_channels=256,
        decoder_output_channels=decoder_channels,
        atrous_rates=[6, 12, 18],
        name='test_decoder')

    input_shapes_list = [[[2, 128, 128, 128], [2, 256, 256, 64],
                          [2, 32, 32, 512]],
                         [[2, 129, 129, 128], [2, 257, 257, 64],
                          [2, 33, 33, 512]]]

    for shapes in input_shapes_list:
      input_dict = {'low_one': tf.random.uniform(shape=shapes[0]),
                    'low_two': tf.random.uniform(shape=shapes[1]),
                    'high': tf.random.uniform(shape=shapes[2])}

      expected_shape = _create_expected_shape(shapes[1], decoder_channels)

      resulting_tensor = decoder(input_dict)
      self.assertListEqual(resulting_tensor.shape.as_list(), expected_shape)

  def test_panoptic_deeplab_single_head_output_shape(self):
    output_channels = 19
    head = panoptic_deeplab.PanopticDeepLabSingleHead(
        intermediate_channels=256,
        output_channels=output_channels,
        pred_key='pred',
        name='test_head')

    input_shapes_list = [[2, 256, 256, 48], [2, 257, 257, 48]]
    for shape in input_shapes_list:
      input_tensor = tf.random.uniform(shape=shape)
      expected_shape = _create_expected_shape(shape, output_channels)

      resulting_tensor = head(input_tensor)
      self.assertListEqual(resulting_tensor['pred'].shape.as_list(),
                           expected_shape)

  def test_panoptic_deeplab_decoder_output_shape(self):
    num_classes = 31
    model_options = _create_panoptic_deeplab_example_proto(
        num_classes=num_classes)
    decoder = panoptic_deeplab.PanopticDeepLab(
        panoptic_deeplab_options=model_options.panoptic_deeplab,
        decoder_options=model_options.decoder)

    input_shapes_list = [[[2, 256, 256, 64], [2, 128, 128, 128],
                          [2, 32, 32, 512]],
                         [[2, 257, 257, 64], [2, 129, 129, 128],
                          [2, 33, 33, 512]]]

    for shapes in input_shapes_list:
      input_dict = {'res2': tf.random.uniform(shape=shapes[0]),
                    'res3': tf.random.uniform(shape=shapes[1]),
                    'res5': tf.random.uniform(shape=shapes[2])}

      expected_semantic_shape = _create_expected_shape(shapes[0], num_classes)
      expected_instance_center_shape = _create_expected_shape(shapes[0], 1)
      expected_instance_regression_shape = _create_expected_shape(shapes[0], 2)

      resulting_dict = decoder(input_dict)
      self.assertListEqual(
          resulting_dict[common.PRED_SEMANTIC_LOGITS_KEY].shape.as_list(),
          expected_semantic_shape)
      self.assertListEqual(
          resulting_dict[common.PRED_CENTER_HEATMAP_KEY].shape.as_list(),
          expected_instance_center_shape)
      self.assertListEqual(
          resulting_dict[common.PRED_OFFSET_MAP_KEY].shape.as_list(),
          expected_instance_regression_shape)

  @test_utils.test_all_strategies
  def test_panoptic_deeplab_sync_bn(self, strategy):
    num_classes = 31
    model_options = _create_panoptic_deeplab_example_proto(
        num_classes=num_classes)
    input_dict = {'res2': tf.random.uniform(shape=[2, 257, 257, 64]),
                  'res3': tf.random.uniform(shape=[2, 129, 129, 128]),
                  'res5': tf.random.uniform(shape=[2, 33, 33, 512])}

    with strategy.scope():
      for bn_layer in test_utils.NORMALIZATION_LAYERS:
        decoder = panoptic_deeplab.PanopticDeepLab(
            panoptic_deeplab_options=model_options.panoptic_deeplab,
            decoder_options=model_options.decoder,
            bn_layer=bn_layer)
        _ = decoder(input_dict)

  def test_panoptic_deeplab_single_decoder_logging_feature_order(self):
    with self.assertLogs(level='WARN'):
      _ = panoptic_deeplab.PanopticDeepLabSingleDecoder(
          high_level_feature_name='high',
          low_level_feature_names=['low_two', 'low_one'],
          low_level_channels_project=[32, 64],  # Potentially wrong order.
          aspp_output_channels=256,
          decoder_output_channels=256,
          atrous_rates=[6, 12, 18],
          name='test_decoder')

  def test_panoptic_deeplab_decoder_ckpt_tems(self):
    num_classes = 31
    model_options = _create_panoptic_deeplab_example_proto(
        num_classes=num_classes)
    decoder = panoptic_deeplab.PanopticDeepLab(
        panoptic_deeplab_options=model_options.panoptic_deeplab,
        decoder_options=model_options.decoder)
    ckpt_dict = decoder.checkpoint_items
    self.assertIn(common.CKPT_SEMANTIC_DECODER, ckpt_dict)
    self.assertIn(common.CKPT_SEMANTIC_HEAD_WITHOUT_LAST_LAYER, ckpt_dict)
    self.assertIn(common.CKPT_SEMANTIC_LAST_LAYER, ckpt_dict)
    self.assertIn(common.CKPT_INSTANCE_DECODER, ckpt_dict)
    self.assertIn(common.CKPT_INSTANCE_REGRESSION_HEAD_WITHOUT_LAST_LAYER,
                  ckpt_dict)
    self.assertIn(common.CKPT_INSTANCE_REGRESSION_HEAD_LAST_LAYER, ckpt_dict)
    self.assertIn(common.CKPT_INSTANCE_CENTER_HEAD_WITHOUT_LAST_LAYER,
                  ckpt_dict)
    self.assertIn(common.CKPT_INSTANCE_CENTER_HEAD_LAST_LAYER, ckpt_dict)


if __name__ == '__main__':
  tf.test.main()
