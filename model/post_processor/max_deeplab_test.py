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

"""Test for max_deeplab.py."""
import numpy as np
import tensorflow as tf

from deeplab2.model.post_processor import max_deeplab


class PostProcessingTest(tf.test.TestCase):

  def test_filter_by_count(self):
    input_index_map = tf.convert_to_tensor(
        [[[1, 1, 1, 1],
          [1, 2, 2, 1],
          [3, 3, 3, 3],
          [4, 5, 5, 5]],
         [[4, 5, 5, 5],
          [3, 3, 3, 3],
          [1, 2, 2, 1],
          [1, 1, 1, 1]]], dtype=tf.float32)
    area_limit = 3
    filtered_index_map, mask = max_deeplab._filter_by_count(
        input_index_map, area_limit)

    expected_filtered_index_map = tf.convert_to_tensor(
        [[[1, 1, 1, 1],
          [1, 0, 0, 1],
          [3, 3, 3, 3],
          [0, 5, 5, 5]],
         [[0, 5, 5, 5],
          [3, 3, 3, 3],
          [1, 0, 0, 1],
          [1, 1, 1, 1]]], dtype=tf.float32)
    np.testing.assert_equal(filtered_index_map.numpy(),
                            expected_filtered_index_map.numpy())
    expected_mask = tf.convert_to_tensor(
        [[[1, 1, 1, 1],
          [1, 0, 0, 1],
          [1, 1, 1, 1],
          [0, 1, 1, 1]],
         [[0, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 0, 0, 1],
          [1, 1, 1, 1]]], dtype=tf.float32)
    np.testing.assert_equal(mask.numpy(), expected_mask.numpy())

  def test_get_mask_id_and_semantic_maps(self):
    height = 21
    width = 21
    num_mask_slots = 5
    num_thing_stuff_classes = 19
    thing_class_ids = list(range(11, 19))
    stuff_class_ids = list(range(0, 11))
    pixel_space_mask_logits = tf.random.uniform(
        (height, width, num_mask_slots), minval=-10, maxval=10)
    # Class scores are normalized beforehand (softmax-ed beforehand).
    transformer_class_probs = tf.random.uniform(
        (num_mask_slots, num_thing_stuff_classes + 1), minval=0, maxval=1)
    input_shape = [41, 41]
    pixel_confidence_threshold = 0.4
    transformer_class_confidence_threshold = 0.7
    pieces = 2

    mask_id_map, semantic_map, thing_mask, stuff_mask = (
        max_deeplab._get_mask_id_and_semantic_maps(
            thing_class_ids, stuff_class_ids, pixel_space_mask_logits,
            transformer_class_probs, input_shape, pixel_confidence_threshold,
            transformer_class_confidence_threshold, pieces)
        )
    self.assertListEqual(mask_id_map.get_shape().as_list(), input_shape)
    self.assertListEqual(semantic_map.get_shape().as_list(), input_shape)
    self.assertListEqual(thing_mask.get_shape().as_list(), input_shape)
    self.assertListEqual(stuff_mask.get_shape().as_list(), input_shape)

  def test_merge_mask_id_and_semantic_maps(self):
    mask_id_maps = tf.convert_to_tensor(
        [[[1, 1, 1, 1],
          [1, 2, 2, 1],
          [3, 3, 4, 4],
          [5, 5, 6, 6]]], dtype=tf.int32)
    semantic_maps = tf.convert_to_tensor(
        [[[0, 0, 0, 0],
          [0, 1, 1, 0],
          [2, 2, 2, 2],
          [2, 2, 3, 3]]], dtype=tf.int32)
    thing_masks = tf.convert_to_tensor(
        [[[0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [1, 0, 1, 1]]], dtype=tf.float32)  # thing_class_ids = [2, 3]
    stuff_masks = tf.convert_to_tensor(
        [[[1, 1, 1, 0],
          [1, 1, 1, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]], dtype=tf.float32)  # stuff_class_ids = [0, 1]

    batch_size = 3
    mask_id_maps = tf.repeat(mask_id_maps, repeats=batch_size, axis=0)
    semantic_maps = tf.repeat(semantic_maps, repeats=batch_size, axis=0)
    thing_masks = tf.repeat(thing_masks, repeats=batch_size, axis=0)
    stuff_masks = tf.repeat(stuff_masks, repeats=batch_size, axis=0)

    label_divisor = 100
    stuff_area_limit = 3
    void_label = 255
    thing_area_limit = 2
    # The expected_panoptic_prediction is computed as follows.
    # All un-certain regions will be labeled as `void_label * label_divisor`.
    # For `thing` segmentation, instance 3, 4, and 6 are kept, but instance 5
    # is re-labeled as `void_label * label_divisor` since its area had been
    # reduced by `confident_regions` and is then filtered by thing_area_limit.
    # For `stuff` segmentation, class-0 region is kept, while class-1 region
    # is re-labeled as `void_label * label_divisor` since its area is smaller
    # than stuff_area_limit.
    expected_panoptic_prediction = tf.convert_to_tensor(
        [[[0, 0, 0, void_label * label_divisor],
          [0, void_label * label_divisor, void_label * label_divisor, 0],
          [2 * label_divisor + 3, 2 * label_divisor + 3, 2 * label_divisor + 4,
           2 * label_divisor + 4],
          [void_label * label_divisor, void_label * label_divisor,
           3 * label_divisor + 6, 3 * label_divisor + 6]]],
        dtype=tf.int32)
    expected_panoptic_prediction = tf.repeat(
        expected_panoptic_prediction, repeats=batch_size, axis=0)
    panoptic_prediction = (
        max_deeplab._merge_mask_id_and_semantic_maps(
            mask_id_maps, semantic_maps, thing_masks, stuff_masks, void_label,
            label_divisor, thing_area_limit, stuff_area_limit))

    np.testing.assert_equal(expected_panoptic_prediction.numpy(),
                            panoptic_prediction.numpy())

  def test_get_panoptic_predictions(self):
    batch = 1
    height = 5
    width = 5
    num_thing_stuff_classes = 2
    thing_class_ids = list(range(1, num_thing_stuff_classes + 1))  # [1, 2]
    label_divisor = 10
    stuff_area_limit = 3
    void_label = 0  # `class-0` is `void`

    o, x = 10, -10
    pixel_space_mask_logits = tf.convert_to_tensor(
        [[[[o, o, o, o, o],  # instance-1 mask
           [o, x, x, o, o],
           [x, x, x, x, x],
           [x, x, x, x, x],
           [x, x, x, x, x]],

          [[x, x, x, x, x],  # instance-2 mask
           [x, o, o, x, x],
           [x, o, o, x, x],
           [x, o, o, x, x],
           [x, x, x, x, x]],

          [[x, x, x, x, x],  # instance-3 mask
           [x, x, x, x, x],
           [o, x, x, o, o],
           [o, x, x, o, o],
           [o, o, o, o, o]]]],
        dtype=tf.float32)
    pixel_space_mask_logits = tf.transpose(pixel_space_mask_logits,
                                           perm=[0, 2, 3, 1])  # b, h, w, c
    # class scores are 0-1 normalized beforehand.
    # 3-rd column (class-2) represents `void` class scores.
    transformer_class_logits = tf.convert_to_tensor(
        [[
            [o, x, x],  # instance-1 -- class-0
            [o, x, x],  # instance-2 -- class-0
            [x, o, x],  # instance-3 -- class-1
        ]], dtype=tf.float32)

    input_shape = [5, 5]
    pixel_confidence_threshold = 0.4
    transformer_class_confidence_threshold = 0.7
    thing_area_limit = 3
    pieces = 1  # No piece-wise operation used.

    panoptic_maps, mask_id_maps, semantic_maps = (
        max_deeplab._get_panoptic_predictions(
            pixel_space_mask_logits, transformer_class_logits, thing_class_ids,
            void_label, label_divisor, thing_area_limit, stuff_area_limit,
            input_shape, pixel_confidence_threshold,
            transformer_class_confidence_threshold, pieces)
        )
    self.assertSequenceEqual(panoptic_maps.shape, (batch, height, width))
    self.assertSequenceEqual(semantic_maps.shape, (batch, height, width))
    self.assertSequenceEqual(mask_id_maps.shape, (batch, height, width))
    expected_panoptic_maps = [[  # label_divisor = 10
        [11, 11, 11, 11, 11],  # 11: semantic_id=1, instance_id=1
        [11, 12, 12, 11, 11],  # 12: semantic_id=1, instance_id=2
        [23, 12, 12, 23, 23],  # 23: semantic_id=2, instance_id=3
        [23, 12, 12, 23, 23],
        [23, 23, 23, 23, 23],
    ]]
    np.testing.assert_array_equal(panoptic_maps, expected_panoptic_maps)
    expected_mask_id_maps = [[
        [1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1],
        [3, 2, 2, 3, 3],
        [3, 2, 2, 3, 3],
        [3, 3, 3, 3, 3],
    ]]
    np.testing.assert_array_equal(mask_id_maps, expected_mask_id_maps)
    expected_semantic_maps = [[
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [2, 1, 1, 2, 2],
        [2, 1, 1, 2, 2],
        [2, 2, 2, 2, 2],
    ]]
    np.testing.assert_array_equal(semantic_maps, expected_semantic_maps)


if __name__ == '__main__':
  tf.test.main()
