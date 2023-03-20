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

"""Test for drop_path.py."""
import numpy as np
import tensorflow as tf

from deeplab2.model.layers import drop_path

# Set a fixed random seed.
tf.random.set_seed(1)


class DropPathTest(tf.test.TestCase):

  def test_drop_path_keep_prob_one(self):
    # Test drop_path_keep_prob = 1, where output should be equal to input.
    drop_path_keep_prob = 1.0
    input_tensor = tf.random.uniform(shape=(3, 65, 65, 32))
    layer_op = drop_path.DropPath(drop_path_keep_prob)
    output_tensor = layer_op(input_tensor, training=True)
    np.testing.assert_equal(input_tensor.numpy(), output_tensor.numpy())

  def test_not_training_mode(self):
    # Test not training mode, where output should be equal to input.
    drop_path_keep_prob = 0.8
    input_tensor = tf.random.uniform(shape=(3, 65, 65, 32))
    layer_op = drop_path.DropPath(drop_path_keep_prob)
    output_tensor = layer_op(input_tensor, training=False)
    np.testing.assert_equal(input_tensor.numpy(), output_tensor.numpy())

  def test_drop_path(self):
    drop_path_keep_prob = 0.8
    input_tensor = tf.random.uniform(shape=(3, 65, 65, 32))
    layer_op = drop_path.DropPath(drop_path_keep_prob)
    output_tensor = layer_op(input_tensor, training=True)
    self.assertFalse(np.array_equal(input_tensor.numpy(),
                                    output_tensor.numpy()))

  def test_constant_drop_path_schedule(self):
    keep_prob_for_last_stage = 0.8
    current_stage_keep_prob = drop_path.get_drop_path_keep_prob(
        keep_prob_for_last_stage,
        schedule='constant',
        current_stage=2,
        num_stages=5)
    self.assertEqual(current_stage_keep_prob, keep_prob_for_last_stage)

  def test_linear_drop_path_schedule(self):
    keep_prob_for_last_stage = 0.8
    current_stage_keep_prob = drop_path.get_drop_path_keep_prob(
        keep_prob_for_last_stage,
        schedule='linear',
        current_stage=1,
        num_stages=4)
    self.assertEqual(current_stage_keep_prob, 0.95)

  def test_unknown_drop_path_schedule(self):
    with self.assertRaises(ValueError):
      _ = drop_path.get_drop_path_keep_prob(0.8, 'unknown', 1, 4)


if __name__ == '__main__':
  tf.test.main()
