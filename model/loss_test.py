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

"""Tests for loss."""

import numpy as np
import tensorflow as tf

from deeplab2 import common
from deeplab2 import trainer_pb2
from deeplab2.model import loss


class LossTest(tf.test.TestCase):

  def test_general_loss(self):
    y_true = {
        'gt': tf.ones([2, 33, 33]) * 2,
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'target': tf.zeros([2, 33, 33])}

    with self.subTest('L1'):
      loss_layer = loss.TopKGeneralLoss(
          loss.mean_absolute_error,
          'gt',
          'target',
          'weight')
      expected_loss = tf.ones([2]) * 2
    with self.subTest('MSE'):
      loss_layer = loss.TopKGeneralLoss(
          loss.mean_squared_error,
          'gt',
          'target',
          'weight')
      expected_loss = tf.ones([2]) * 4
    loss_result = loss_layer(y_true, y_pred)
    np.testing.assert_almost_equal(
        loss_result.numpy(), expected_loss.numpy(), decimal=5)

  def test_general_loss_weights(self):
    weights = np.zeros((2, 33, 33))
    weights[:, 17:29, 15:23] = 1

    gt = np.ones([2, 33, 33]) * 1.5
    gt[:, 17:29, 15:23] = 2

    y_true = {
        'gt': tf.convert_to_tensor(gt, dtype=tf.float32),
        'weight': tf.convert_to_tensor(weights, dtype=tf.float32)
    }
    y_pred = {'target': tf.zeros([2, 33, 33])}
    loss_layer = loss.TopKGeneralLoss(
        loss.mean_absolute_error,
        'gt',
        'target',
        'weight')

    expected_loss = tf.ones([2]) * 2
    loss_result = loss_layer(y_true, y_pred)

    np.testing.assert_almost_equal(
        loss_result.numpy(), expected_loss.numpy(), decimal=5)

  def test_topk_ce_loss_ignore(self):
    num_classes = 19
    ignore_label = 255
    loss_layer = loss.TopKCrossEntropyLoss(
        gt_key='gt',
        target_key='target',
        weight_key='weight',
        num_classes=num_classes,
        ignore_label=ignore_label)

    gt_tensor = np.ones(shape=[2, 33, 33], dtype=np.int32) * ignore_label
    gt_tensor[:, 17:29, 15:23] = 1
    logits = tf.random.uniform(shape=[2, 33, 33, num_classes])

    y_true = {
        'gt': tf.convert_to_tensor(gt_tensor),
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'target': logits}

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(np.squeeze(gt_tensor[:, 17:29, 15:23]), num_classes),
        logits[:, 17:29, 15:23, :])
    expected_result = tf.reduce_mean(expected_result, axis=[1, 2])

    per_sample_loss = loss_layer(y_true, y_pred)

    np.testing.assert_almost_equal(
        per_sample_loss.numpy(), expected_result.numpy(), decimal=5)

  def test_topk_ce_loss_global_weight(self):
    num_classes = 19
    weight = 3.145
    loss_layer = loss.TopKCrossEntropyLoss(
        gt_key='gt',
        target_key='target',
        weight_key='weight',
        num_classes=num_classes)
    logits = tf.random.uniform(shape=[2, 33, 33, num_classes])

    y_true = {
        'gt': tf.ones([2, 33, 33], tf.int32),
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'target': logits}

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(y_true['gt'], num_classes), logits)
    expected_result = tf.reduce_mean(expected_result, axis=[1, 2])
    expected_result *= weight

    per_sample_loss = loss_layer(y_true, y_pred, weight)

    np.testing.assert_almost_equal(
        per_sample_loss.numpy(), expected_result.numpy(), decimal=5)

  def test_topk_ce_loss_topk(self):
    num_classes = 19
    top_k = 0.5
    loss_layer = loss.TopKCrossEntropyLoss(
        gt_key='gt',
        target_key='target',
        weight_key='weight',
        num_classes=num_classes,
        top_k_percent_pixels=top_k)

    logits = tf.random.uniform(shape=[2, 33, 33, num_classes])
    y_true = {
        'gt': tf.ones([2, 33, 33], tf.int32),
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'target': logits}

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(y_true['gt'], num_classes), logits)
    expected_result, _ = tf.math.top_k(
        tf.reshape(expected_result, shape=[2, -1]),
        tf.cast((top_k * tf.size(y_true['gt'], tf.float32) / 2), tf.int32))
    expected_result = tf.reduce_mean(expected_result, axis=[1])

    per_sample_loss = loss_layer(y_true, y_pred)

    np.testing.assert_almost_equal(
        per_sample_loss.numpy(), expected_result.numpy(), decimal=5)

  def test_panoptic_deeplab_loss(self):
    ignore_label = 255
    num_classes = 19
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    center_loss_options = trainer_pb2.LossOptions.SingleLossOptions(name='mse')
    regression_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1')
    motion_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1')

    loss_layer = loss.DeepLabFamilyLoss(
        semantic_loss_options,
        center_loss_options,
        regression_loss_options,
        motion_loss_options=motion_loss_options,
        num_classes=num_classes,
        ignore_label=ignore_label)

    pred_dict = {
        common.TARGET_SEMANTIC_LOGITS_KEY:
            tf.random.uniform(shape=[2, 33, 33, num_classes]),
        common.TARGET_CENTER_HEATMAP_KEY:
            tf.zeros(shape=[2, 33, 33]),
        common.TARGET_OFFSET_MAP_KEY:
            tf.zeros(shape=[2, 33, 33, 2]),
        common.TARGET_FRAME_OFFSET_MAP_KEY:
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
      loss_result = tf.reduce_sum(loss_layer(gt_dict, pred_dict), axis=1)

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
      loss_result = tf.reduce_sum(loss_layer(gt_dict, pred_dict), axis=1)

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
      loss_result = tf.reduce_sum(loss_layer(gt_dict, pred_dict), axis=1)

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
          pred_dict[common.TARGET_SEMANTIC_LOGITS_KEY])
      expected_result = tf.reduce_mean(expected_result, axis=[1, 2])
      # Add center and regression loss.
      expected_result += tf.ones(shape=[2]) * 8

      loss_result = tf.reduce_sum(loss_layer(gt_dict, pred_dict), axis=1)

      np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

  def test_panoptic_deeplab_semantic_loss_only(self):
    ignore_label = 255
    num_classes = 19
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')

    loss_layer = loss.DeepLabFamilyLoss(
        semantic_loss_options,
        center_loss_options=None,
        regression_loss_options=None,
        motion_loss_options=None,
        num_classes=num_classes,
        ignore_label=ignore_label)

    pred_dict = {
        common.TARGET_SEMANTIC_LOGITS_KEY:
            tf.random.uniform(shape=[2, 33, 33, num_classes]),
    }
    gt_dict = {
        common.GT_SEMANTIC_KEY: tf.ones(shape=[2, 33, 33], dtype=tf.int32),
        common.SEMANTIC_LOSS_WEIGHT_KEY: tf.ones(shape=[2, 33, 33]),
    }

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(gt_dict[common.GT_SEMANTIC_KEY], num_classes),
        pred_dict[common.TARGET_SEMANTIC_LOGITS_KEY])
    expected_result = tf.reduce_mean(expected_result, axis=[1, 2])

    loss_result = tf.reduce_sum(loss_layer(gt_dict, pred_dict), axis=1)

    np.testing.assert_equal(loss_result.numpy(), expected_result.numpy())

  def test_panoptic_deeplab_loss_error(self):
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    center_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='not_a_loss', weight=1.0)
    regression_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1', weight=1.0)
    moyion_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='l1', weight=1.0)

    with self.assertRaises(ValueError):
      _ = loss.DeepLabFamilyLoss(
          semantic_loss_options,
          center_loss_options,
          regression_loss_options,
          moyion_loss_options,
          num_classes=19)


if __name__ == '__main__':
  tf.test.main()
