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

"""Tests for utils."""

import itertools

import numpy as np
import tensorflow as tf

from deeplab2.model import utils


class UtilsTest(tf.test.TestCase):

  def test_resize_logits_graph_mode(self):
    @tf.function
    def graph_mode_wrapper(*args):
      return utils.resize_and_rescale_offsets(*args)

    resized_logits = graph_mode_wrapper(tf.ones((2, 33, 33, 2)), [65, 65])
    resized_logits_2 = graph_mode_wrapper(tf.ones((2, 33, 33, 2)), [33, 33])
    self.assertListEqual(resized_logits.shape.as_list(), [2, 65, 65, 2])
    self.assertListEqual(resized_logits_2.shape.as_list(), [2, 33, 33, 2])

  def test_resize_logits(self):
    offset_logits = tf.convert_to_tensor([[[[2, 2], [2, 1], [2, 0]],
                                           [[1, 2], [1, 1], [1, 0]],
                                           [[0, 2], [0, 1], [0, 0]]]],
                                         dtype=tf.float32)
    target_size = [5, 5]
    resized_logits = utils.resize_and_rescale_offsets(offset_logits,
                                                      target_size)

    self.assertListEqual(resized_logits.shape.as_list(), [1, 5, 5, 2])
    for i in range(5):
      for j in range(5):
        np.testing.assert_array_almost_equal(resized_logits.numpy()[0, i, j, :],
                                             [4 - i, 4 - j])

  def test_zero_padding(self):
    input_tensor = tf.ones(shape=(2, 5, 5, 2))
    input_tensor_2 = tf.ones(shape=(5, 5, 2))
    padded_tensor = utils.add_zero_padding(input_tensor, kernel_size=5, rank=4)
    padded_tensor_2 = utils.add_zero_padding(
        input_tensor_2, kernel_size=5, rank=3)

    self.assertEqual(tf.reduce_sum(padded_tensor), 100)
    self.assertEqual(tf.reduce_sum(padded_tensor_2), 50)
    self.assertListEqual(padded_tensor.shape.as_list(), [2, 9, 9, 2])
    self.assertListEqual(padded_tensor_2.shape.as_list(), [9, 9, 2])
    # Count zero elements.
    self.assertEqual(tf.reduce_sum(padded_tensor-1), -224)
    self.assertEqual(tf.reduce_sum(padded_tensor_2-1), -112)

  def test_resize_function_error(self):
    input_tensor = tf.random.uniform(shape=(2, 10, 10, 2))
    with self.assertRaises(ValueError):
      _ = utils.resize_align_corners(input_tensor, [19, 19],
                                     method='not_a_valid_method')

  def test_resize_function_shape(self):
    input_tensor = tf.random.uniform(shape=(2, 10, 10, 2))
    result_tensor = utils.resize_align_corners(input_tensor, [19, 19])

    self.assertListEqual(result_tensor.shape.as_list(), [2, 19, 19, 2])

  def test_resize_graph_mode(self):
    @tf.function
    def graph_mode_wrapper(*args):
      return utils.resize_align_corners(*args)

    result_tensor = graph_mode_wrapper(tf.ones((2, 33, 33, 2)), [65, 65])
    result_tensor_2 = graph_mode_wrapper(tf.ones((2, 33, 33, 2)), [33, 33])
    self.assertListEqual(result_tensor.shape.as_list(), [2, 65, 65, 2])
    self.assertListEqual(result_tensor_2.shape.as_list(), [2, 33, 33, 2])

  def test_resize_function_constant_input(self):
    input_tensor = tf.ones(shape=(2, 10, 10, 2))
    result_tensor = utils.resize_align_corners(input_tensor, [19, 19])

    self.assertTrue(tf.keras.backend.all(result_tensor == 1))

  def test_resize_function_invalid_rank(self):
    input_tensor = tf.keras.Input(shape=(None, 2))
    with self.assertRaisesRegex(
        ValueError, 'should have rank of 4'):
      _ = utils.resize_align_corners(input_tensor, [19, 19])

  def test_resize_function_v1_compatibility(self):
    # Test for odd and even input, and output shapes.
    input_shapes = [(2, 10, 10, 3), (2, 11, 11, 3)]
    target_sizes = [[19, 19], [20, 20]]
    methods = ['bilinear', 'nearest']

    for shape, target_size, method in itertools.product(input_shapes,
                                                        target_sizes, methods):
      input_tensor = tf.random.uniform(shape=shape)

      result_tensor = utils.resize_align_corners(input_tensor, target_size,
                                                 method)
      if method == 'bilinear':
        expected_tensor = tf.compat.v1.image.resize(
            input_tensor,
            target_size,
            align_corners=True,
            method=tf.compat.v1.image.ResizeMethod.BILINEAR)
      else:
        expected_tensor = tf.compat.v1.image.resize(
            input_tensor,
            target_size,
            align_corners=True,
            method=tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

      np.testing.assert_equal(result_tensor.numpy(), expected_tensor.numpy())

  def test_resize_bilinear_v1_compatibility(self):
    # Test for odd and even input, and output shapes.
    input_shapes = [(2, 10, 10, 3), (2, 11, 11, 3), (1, 11, 11, 64)]
    target_sizes = [[19, 19], [20, 20], [10, 10]]

    for shape, target_size in itertools.product(input_shapes, target_sizes):
      input_tensor = tf.random.uniform(shape=shape)
      result_tensor = utils.resize_bilinear(input_tensor, target_size)
      expected_tensor = tf.compat.v1.image.resize(
          input_tensor,
          target_size,
          align_corners=True,
          method=tf.compat.v1.image.ResizeMethod.BILINEAR)
      self.assertAllClose(result_tensor, expected_tensor)

  def test_make_divisible(self):
    value, divisor, min_value = 17, 2, 8
    new_value = utils.make_divisible(value, divisor, min_value)
    self.assertAllEqual(new_value, 18)

    value, divisor, min_value = 17, 2, 22
    new_value = utils.make_divisible(value, divisor, min_value)
    self.assertAllEqual(new_value, 22)

  def test_transpose_and_reshape_for_attention_operation(self):
    images = tf.zeros([2, 8, 11, 2])
    output = utils.transpose_and_reshape_for_attention_operation(images)
    self.assertEqual(output.get_shape().as_list(), [2, 11, 16])

  def test_reshape_and_transpose_for_attention_operation(self):
    images = tf.zeros([2, 11, 16])
    output = utils.reshape_and_transpose_for_attention_operation(images,
                                                                 num_heads=8)
    self.assertEqual(output.get_shape().as_list(), [2, 8, 11, 2])

  def test_safe_setattr_raise_error(self):
    layer = tf.keras.layers.Conv2D(1, 1)
    with self.assertRaises(ValueError):
      utils.safe_setattr(layer, 'filters', 3)

    utils.safe_setattr(layer, 'another_conv', tf.keras.layers.Conv2D(1, 1))
    with self.assertRaises(ValueError):
      utils.safe_setattr(layer, 'another_conv', tf.keras.layers.Conv2D(1, 1))

  def test_pad_sequence_with_none(self):
    sequence = [1, 2]
    output_2 = utils.pad_sequence_with_none(sequence, target_length=2)
    self.assertEqual(output_2, [1, 2])
    output_3 = utils.pad_sequence_with_none(sequence, target_length=3)
    self.assertEqual(output_3, [1, 2, None])

  def test_strided_downsample(self):
    inputs = tf.zeros([2, 11, 11])
    output = utils.strided_downsample(inputs, target_size=[6, 6])
    self.assertEqual(output.get_shape().as_list(), [2, 6, 6])

  def test_get_stuff_class_ids(self):
    # num_thing_stuff_classes does not include `void` class.
    num_thing_stuff_classes = 5
    thing_class_ids = [3, 4]
    void_label_list = [5, 0]
    expected_stuff_class_ids_list = [
        [0, 1, 2], [1, 2, 5]
    ]
    for void_label, expected_stuff_class_ids in zip(
        void_label_list, expected_stuff_class_ids_list):
      stuff_class_ids = utils.get_stuff_class_ids(
          num_thing_stuff_classes, thing_class_ids, void_label)
      np.testing.assert_equal(stuff_class_ids,
                              expected_stuff_class_ids)

if __name__ == '__main__':
  tf.test.main()
