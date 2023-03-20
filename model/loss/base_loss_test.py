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

"""Tests for base_loss.py."""

import numpy as np
import tensorflow as tf

from deeplab2.model.loss import base_loss as loss


class BaseLossTest(tf.test.TestCase):

  def test_general_loss(self):
    y_true = {
        'gt': tf.ones([2, 33, 33]) * 2,
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'pred': tf.zeros([2, 33, 33])}

    with self.subTest('L1'):
      loss_layer = loss.TopKGeneralLoss(
          loss.mean_absolute_error,
          'gt',
          'pred',
          'weight')
      expected_loss = tf.ones([2]) * 2
    with self.subTest('MSE'):
      loss_layer = loss.TopKGeneralLoss(
          loss.mean_squared_error,
          'gt',
          'pred',
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
    y_pred = {'pred': tf.zeros([2, 33, 33])}
    loss_layer = loss.TopKGeneralLoss(
        loss.mean_absolute_error,
        'gt',
        'pred',
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
        pred_key='pred',
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
    y_pred = {'pred': logits}

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
        pred_key='pred',
        weight_key='weight',
        num_classes=num_classes,
        ignore_label=255)
    logits = tf.random.uniform(shape=[2, 33, 33, num_classes])

    y_true = {
        'gt': tf.ones([2, 33, 33], tf.int32),
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'pred': logits}

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
        pred_key='pred',
        weight_key='weight',
        num_classes=num_classes,
        top_k_percent_pixels=top_k,
        ignore_label=255)

    logits = tf.random.uniform(shape=[2, 33, 33, num_classes])
    y_true = {
        'gt': tf.ones([2, 33, 33], tf.int32),
        'weight': tf.ones([2, 33, 33])
    }
    y_pred = {'pred': logits}

    expected_result = tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(y_true['gt'], num_classes), logits)
    expected_result, _ = tf.math.top_k(
        tf.reshape(expected_result, shape=[2, -1]),
        tf.cast((top_k * tf.size(y_true['gt'], tf.float32) / 2), tf.int32))
    expected_result = tf.reduce_mean(expected_result, axis=[1])

    per_sample_loss = loss_layer(y_true, y_pred)

    np.testing.assert_almost_equal(
        per_sample_loss.numpy(), expected_result.numpy(), decimal=5)

  def test_is_one_hot(self):
    num_classes = 19
    gt_list = [
        tf.ones([2, 33, 33], tf.int32),
        tf.ones([2, 33], tf.int32),
        tf.one_hot(tf.ones([2, 33, 33], tf.int32), num_classes),
        tf.one_hot(tf.ones([2, 33], tf.int32), num_classes),
    ]
    pred_list = [
        tf.random.uniform(shape=[2, 33, 33, num_classes]),
        tf.random.uniform(shape=[2, 33, num_classes]),
        tf.random.uniform(shape=[2, 33, 33, num_classes]),
        tf.random.uniform(shape=[2, 33, num_classes]),
    ]
    expected_result_list = [False, False, True, True]
    output_list = []
    for gt, pred in zip(gt_list, pred_list):
      output_list.append(loss.is_one_hot(gt, pred))
    np.testing.assert_equal(output_list, expected_result_list)

  def test_focal_ce_loss_integer_or_one_hot(self):
    num_classes = 19
    gamma = 0.5
    alpha = 0.75
    loss_layer = loss.FocalCrossEntropyLoss(
        gt_key='gt',
        pred_key='pred',
        weight_key='weight',
        num_classes=num_classes,
        focal_loss_alpha=alpha,
        focal_loss_gamma=gamma,
        ignore_label=255)

    logits = tf.random.uniform(shape=[2, 33 * 33, num_classes])
    gt = tf.ones([2, 33 * 33], tf.int32)
    use_one_hot_encode_list = [False, True]
    for use_one_hot_encode in use_one_hot_encode_list:
      if use_one_hot_encode:
        gt = tf.one_hot(gt, num_classes)
      y_true = {'gt': gt}
      y_pred = {'pred': logits,
                'weight': tf.ones([2, 33 * 33])}
      predictions = tf.nn.softmax(logits, axis=-1)
      if use_one_hot_encode:
        pt = tf.reduce_sum(predictions * gt, axis=-1)
        expected_result = tf.nn.softmax_cross_entropy_with_logits(gt, logits)
      else:
        pt = tf.reduce_sum(predictions * tf.one_hot(gt, num_classes), axis=-1)
        expected_result = tf.nn.softmax_cross_entropy_with_logits(
            tf.one_hot(gt, num_classes), logits)
      expected_result = tf.multiply(tf.pow(1.0 - pt, gamma), expected_result)
      expected_result = tf.reshape(expected_result, shape=[2, -1])
      # Since labels has no '19' (background) in this example, only alpha is
      # multiplied.
      expected_result = tf.reduce_mean(expected_result, axis=[1]) * alpha
      per_sample_loss = loss_layer(y_true, y_pred)

      np.testing.assert_almost_equal(
          per_sample_loss.numpy(), expected_result.numpy(), decimal=5)

  def test_mask_dice_loss(self):
    gt = [
        [
            [1., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
        ],
        [
            [0., 0., 0.],
            [1., 1., 1.],
            [1., 1., 1.],
        ],
    ]
    gt = tf.constant(gt, dtype=tf.float32)
    gt = tf.expand_dims(gt, -1)
    gt = tf.transpose(gt, perm=[3, 1, 2, 0])

    y_true = {'gt': gt}

    pred = [
        [
            [1., 1., 0.],
            [1., 1., 0.],
            [1., 1., 0.],
        ],
        [
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
        ],
    ]
    # Multiply 100 to make its Softmax output have 0 or 1 values.
    pred = tf.constant(pred, dtype=tf.float32) * 100.
    pred = tf.expand_dims(pred, -1)
    pred = tf.transpose(pred, perm=[3, 1, 2, 0])
    y_pred = {
        'pred': pred,
        'weight': tf.ones([1]) * 0.5
    }

    loss_layer = loss.MaskDiceLoss(
        gt_key='gt',
        pred_key='pred',
        weight_key='weight',
        prediction_activation='softmax')
    dice_loss = loss_layer(y_true, y_pred)
    loss_result = dice_loss.numpy()
    # For each channel,
    #   nominator = 2 * intersection(=2) + smooth(=1) = 5
    #   denominator = 9 + smooth(=1) = 10
    # Channel-wise sum: [5/10, 5/10] -> [1.0]
    # Weighted result: [1.0] * weight(=0.5) = 0.5
    expected_result = np.array([0.5])
    np.testing.assert_almost_equal(loss_result, expected_result)


if __name__ == '__main__':
  tf.test.main()
