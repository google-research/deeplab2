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

"""Tests for loss_builder.py."""

import numpy as np
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2 import trainer_pb2
from deeplab2.model.loss import loss_builder as loss


class LossTest(tf.test.TestCase):

  def test_panoptic_deeplab_loss(self):
    ignore_label = 255
    ignore_depth = 0
    num_classes = 19
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    center_loss_options = trainer_pb2.LossOptions.SingleLossOptions(name='mse')
    regression_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1')
    motion_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1')
    loss_options = trainer_pb2.LossOptions(
        semantic_loss=semantic_loss_options,
        center_loss=center_loss_options,
        regression_loss=regression_loss_options,
        motion_loss=motion_loss_options)

    loss_layer = loss.DeepLabFamilyLoss(
        loss_options,
        deeplab_options=config_pb2.ModelOptions(),
        num_classes=num_classes,
        ignore_label=ignore_label,
        ignore_depth=ignore_depth,
        thing_class_ids=tuple(range(11, 19)))

    pred_dict = {
        common.PRED_SEMANTIC_LOGITS_KEY:
            tf.random.uniform(shape=[2, 33, 33, num_classes]),
        common.PRED_CENTER_HEATMAP_KEY:
            tf.zeros(shape=[2, 33, 33]),
        common.PRED_OFFSET_MAP_KEY:
            tf.zeros(shape=[2, 33, 33, 2]),
        common.PRED_FRAME_OFFSET_MAP_KEY:
            tf.zeros(shape=[2, 33, 33, 2]),
    }

    with self.subTest('Test center loss.'):
      gt_dict = {
          common.GT_SEMANTIC_KEY:
              tf.ones(shape=[2, 33, 33]) * ignore_label,
          common.GT_INSTANCE_CENTER_KEY:
              tf.ones(shape=[2, 33, 33]) * 2,
          common.GT_INSTANCE_REGRESSION_KEY:
              tf.zeros(shape=[2, 33, 33, 2]),
          common.GT_FRAME_OFFSET_KEY:
              tf.zeros(shape=[2, 33, 33, 2]),
          common.SEMANTIC_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.CENTER_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.FRAME_REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
      }
      # expected_result = square(2 - 0).
      expected_result = tf.ones(shape=[2]) * 4
      loss_result = loss_layer(gt_dict, pred_dict)[common.TOTAL_LOSS]

      np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

    with self.subTest('Test regression loss.'):
      gt_dict = {
          common.GT_SEMANTIC_KEY:
              tf.ones(shape=[2, 33, 33]) * ignore_label,
          common.GT_INSTANCE_CENTER_KEY:
              tf.zeros(shape=[2, 33, 33]),
          common.GT_INSTANCE_REGRESSION_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.GT_FRAME_OFFSET_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.SEMANTIC_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.CENTER_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.FRAME_REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
      }
      expected_result = tf.ones(shape=[2]) * 4
      loss_result = loss_layer(gt_dict, pred_dict)[common.TOTAL_LOSS]

      np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

    with self.subTest('Test instances losses.'):
      gt_dict = {
          common.GT_SEMANTIC_KEY:
              tf.ones(shape=[2, 33, 33]) * ignore_label,
          common.GT_INSTANCE_CENTER_KEY:
              tf.ones(shape=[2, 33, 33]) * 2,
          common.GT_INSTANCE_REGRESSION_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.GT_FRAME_OFFSET_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.SEMANTIC_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.CENTER_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.FRAME_REGRESSION_LOSS_WEIGHT_KEY:
              tf.zeros(shape=[2, 33, 33]),
      }
      expected_result = tf.ones(shape=[2]) * 6
      loss_result = loss_layer(gt_dict, pred_dict)[common.TOTAL_LOSS]

      np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

    with self.subTest('Test all losses.'):
      gt_dict = {
          common.GT_SEMANTIC_KEY:
              tf.ones(shape=[2, 33, 33], dtype=tf.int32),
          common.GT_INSTANCE_CENTER_KEY:
              tf.ones(shape=[2, 33, 33]) * 2,
          common.GT_INSTANCE_REGRESSION_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.GT_FRAME_OFFSET_KEY:
              tf.ones(shape=[2, 33, 33, 2]) * 2,
          common.SEMANTIC_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.CENTER_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
          common.FRAME_REGRESSION_LOSS_WEIGHT_KEY:
              tf.ones(shape=[2, 33, 33]),
      }
      expected_result = tf.nn.softmax_cross_entropy_with_logits(
          tf.one_hot(gt_dict[common.GT_SEMANTIC_KEY], num_classes),
          pred_dict[common.PRED_SEMANTIC_LOGITS_KEY])
      expected_result = tf.reduce_mean(expected_result, axis=[1, 2])
      # Add center and regression loss.
      expected_result += tf.ones(shape=[2]) * 8

      loss_result = loss_layer(gt_dict, pred_dict)[common.TOTAL_LOSS]

      np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

  def test_panoptic_deeplab_semantic_loss_only(self):
    ignore_label = 255
    ignore_depth = 0
    num_classes = 19
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    loss_options = trainer_pb2.LossOptions(
        semantic_loss=semantic_loss_options)

    loss_layer = loss.DeepLabFamilyLoss(
        loss_options,
        deeplab_options=config_pb2.ModelOptions(),
        num_classes=num_classes,
        ignore_label=ignore_label,
        ignore_depth=ignore_depth,
        thing_class_ids=tuple(range(11, 19)))

    pred_dict = {
        common.PRED_SEMANTIC_LOGITS_KEY:
            tf.random.uniform(shape=[2, 33, 33, num_classes]),
    }
    gt_dict = {
        common.GT_SEMANTIC_KEY: tf.ones(shape=[2, 33, 33], dtype=tf.int32),
        common.SEMANTIC_LOSS_WEIGHT_KEY: tf.ones(shape=[2, 33, 33]),
    }

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(gt_dict[common.GT_SEMANTIC_KEY], num_classes),
        pred_dict[common.PRED_SEMANTIC_LOGITS_KEY])
    expected_result = tf.reduce_mean(expected_result, axis=[1, 2])

    loss_dict = loss_layer(gt_dict, pred_dict)
    self.assertIn(common.SEMANTIC_LOSS, loss_dict)
    self.assertNotIn(common.CENTER_LOSS, loss_dict)
    self.assertNotIn(common.REGRESSION_LOSS, loss_dict)
    self.assertNotIn(common.MOTION_LOSS, loss_dict)
    loss_result = loss_dict[common.SEMANTIC_LOSS]

    np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

  def test_panoptic_deeplab_loss_error(self):
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    center_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='not_a_loss', weight=1.0)
    regression_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1', weight=1.0)
    motion_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1', weight=1.0)
    loss_options = trainer_pb2.LossOptions(
        semantic_loss=semantic_loss_options,
        center_loss=center_loss_options,
        regression_loss=regression_loss_options,
        motion_loss=motion_loss_options)

    with self.assertRaises(ValueError):
      _ = loss.DeepLabFamilyLoss(loss_options,
                                 deeplab_options=config_pb2.ModelOptions(),
                                 num_classes=19,
                                 ignore_label=255,
                                 ignore_depth=0,
                                 thing_class_ids=tuple(range(11, 19)))


if __name__ == '__main__':
  tf.test.main()
