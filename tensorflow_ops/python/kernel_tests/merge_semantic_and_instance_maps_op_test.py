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

r"""Tests for merge_semantic_and_instance_maps_op."""

import numpy as np
import tensorflow as tf
from deeplab2.tensorflow_ops.python.ops import merge_semantic_and_instance_maps_op


class MergeSemanticAndInstanceMapsOpTest(tf.test.TestCase):

  def testMergeSemanticAndInstanceMaps(self):
    """Test the op with 2 images."""
    batch = 2
    height = 4
    width = 4

    # Create the instance labels.
    instance_maps = np.zeros((batch, height, width), dtype=np.int32)
    instance_maps[0, :, :] = np.array([[0, 2, 1, 0], [0, 1, 1, 0], [2, 0, 1, 2],
                                       [0, 0, 1, 1]])
    instance_maps[1, :, :] = np.array([[1, 2, 3, 1], [0, 2, 1, 3], [0, 2, 2, 0],
                                       [3, 3, 2, 0]])

    # Create the semantic labels.
    # The instances with the instance label equal to 0 and 2 have the same
    # semantic label. The other instances all have different semantic labels.
    semantic_maps = np.zeros((batch, height, width), dtype=np.int32)
    # Instance 0 has 4 pixels predicted as 0 and 3 pixels predicted as 3.
    # Instance 1 has 6 pixels predicted as 1.
    # Instance 2 has 2 pixels predicted as 0 and 1 pixel predicted as 3.
    semantic_maps[0, :, :] = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [3, 3, 1, 0],
                                       [3, 3, 1, 1]])
    # Instance 0 has 3 pixels predicted as 0 and 1 pixel predicted as 3.
    # Instance 1 has 3 pixels predicted as 1.
    # Instance 2 has 3 pixels predicted as 0 and 2 pixels predicted as 2.
    # Instance 3 has 1 pixel predicted as 0 and 3 pixels predicted as 2.
    semantic_maps[1, :, :] = np.array([[1, 0, 2, 1], [0, 0, 1, 2], [0, 2, 2, 3],
                                       [0, 2, 0, 0]])

    # Create the ID list for things.
    thing_ids = [0, 2]

    # Groundtruth semantic segmentation maps after majority voting.
    gt_semantic_maps = np.zeros((batch, height, width), dtype=np.int32)
    gt_semantic_maps[0, :, :] = np.array([[0, 0, 1, 0], [0, 1, 1, 0],
                                          [3, 3, 1, 0], [3, 3, 1, 1]])
    # Instance 2 takes semantic label 0 after majority voting.
    # Instance 3 takes semantic label 2 after majority voting.
    gt_semantic_maps[1, :, :] = np.array([[1, 0, 2, 1], [0, 0, 1, 2],
                                          [0, 0, 0, 3], [2, 2, 0, 0]])
    # Groundtruth instance segmentation maps.
    gt_instance_maps = np.zeros((batch, 2, height, width), dtype=np.int32)

    # There are two cases for gt_instance_maps in batch 1.
    # Case 1:
    # Instance 0 is re-assigned instance label 1.
    # Instance 2 is re-assigned instance label 2.
    gt_instance_maps[0, 0, :, :] = np.array([[1, 2, 0, 1], [1, 0, 0, 1],
                                             [0, 0, 0, 2], [0, 0, 0, 0]])
    # Case 2:
    # Instance 0 is re-assigned instance label 2.
    # Instance 2 is re-assigned instance label 1.
    gt_instance_maps[0, 1, :, :] = np.array([[2, 1, 0, 2], [2, 0, 0, 2],
                                             [0, 0, 0, 1], [0, 0, 0, 0]])
    # There are two cases for gt_instance_maps in batch 2.
    # Case 1:
    # Instance 0 is re-assigned instance label 1.
    # Instance 2 is re-assigned instance label 2.
    # Instance 3 is re-assigned instance label 1.
    gt_instance_maps[1, 0, :, :] = np.array([[0, 2, 1, 0], [1, 2, 0, 1],
                                             [1, 2, 2, 0], [1, 1, 2, 1]])
    # Case 2:
    # Instance 0 is re-assigned instance label 2.
    # Instance 2 is re-assigned instance label 1.
    # Instance 3 is re-assigned instance label 1.
    gt_instance_maps[1, 1, :, :] = np.array([[0, 1, 1, 0], [2, 1, 0, 1],
                                             [2, 1, 1, 0], [1, 1, 1, 2]])
    # Groundtruth parsing maps.
    label_divisor = 256

    # Run the op.
    parsing_maps = (
        merge_semantic_and_instance_maps_op.merge_semantic_and_instance_maps(
            semantic_maps,
            instance_maps,
            thing_ids,
            label_divisor=label_divisor))
    pass_test = False
    for i in range(2):
      for j in range(2):
        current_gt_instance_maps = np.stack(
            [gt_instance_maps[0, i, :, :], gt_instance_maps[1, j, :, :]],
            axis=0)
        gt_parsing_maps = (
            gt_semantic_maps * label_divisor + current_gt_instance_maps)
        if np.array_equal(parsing_maps, gt_parsing_maps):
          pass_test = True
    self.assertTrue(pass_test)

  def testMergeSemanticAndInstanceMapsWithStuffAreaLimit(self):
    batch = 1
    height = 4
    width = 4

    # Create the instance labels.
    instance_maps = np.zeros((batch, height, width), dtype=np.int32)
    instance_maps[0, :, :] = np.array([[0, 0, 0, 0],
                                       [0, 0, 1, 1],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])

    # Create the semantic labels.
    semantic_maps = np.zeros((batch, height, width), dtype=np.int32)
    semantic_maps[0, :, :] = np.array([[0, 0, 0, 0],
                                       [0, 0, 1, 1],
                                       [0, 0, 2, 2],
                                       [0, 0, 2, 2]])
    thing_ids = [0, 2]
    stuff_area_limit = 3
    void_label = 3
    # Groundtruth semantic segmentation maps after majority voting.
    # Instance 0 takes semantic label 0.
    # Instance 1 is re-assigned with void label.
    gt_semantic_maps = np.zeros((batch, height, width), dtype=np.int32)
    gt_semantic_maps[0, :, :] = np.array([[0, 0, 0, 0],
                                          [0, 0, void_label, void_label],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]])
    # Groundtruth instance segmentation maps.
    gt_instance_maps = np.zeros((batch, height, width), dtype=np.int32)
    gt_instance_maps[0, :, :] = np.array([[1, 1, 1, 1],
                                          [1, 1, 0, 0],
                                          [1, 1, 1, 1],
                                          [1, 1, 1, 1]])
    label_divisor = 256
    gt_parsing_maps = gt_semantic_maps * label_divisor + gt_instance_maps

    # Run the op.
    parsing_maps = (
        merge_semantic_and_instance_maps_op.merge_semantic_and_instance_maps(
            semantic_maps,
            instance_maps,
            thing_ids,
            label_divisor=label_divisor,
            stuff_area_limit=stuff_area_limit,
            void_label=void_label))
    self.assertTrue(np.array_equal(parsing_maps, gt_parsing_maps))


class MergeSemanticAndInstanceMapsOpGpuTest(MergeSemanticAndInstanceMapsOpTest):

  def session(self, use_gpu=True):
    return super(MergeSemanticAndInstanceMapsOpGpuTest,
                 self).session(use_gpu=use_gpu)

  def testMergeSemanticAndInstanceMapsWithRandomInputs(self):
    batch = 1
    height = 1441
    width = 1441
    rng = np.random.RandomState(0)
    instance_maps = rng.randint(0, 255, (batch, height, width), dtype=np.int32)
    semantic_maps = rng.randint(0, 3, (batch, height, width), dtype=np.int32)

    thing_ids = [0, 2]
    stuff_area_limit = 400
    void_label = 3
    label_divisor = 256

    with self.session(use_gpu=False):
      parsing_maps_cpu = (
          merge_semantic_and_instance_maps_op.merge_semantic_and_instance_maps(
              semantic_maps,
              instance_maps,
              thing_ids,
              label_divisor=label_divisor,
              stuff_area_limit=stuff_area_limit,
              void_label=void_label))
      parsing_maps_cpu = parsing_maps_cpu.numpy()

    with self.session():
      parsing_maps_gpu = (
          merge_semantic_and_instance_maps_op.merge_semantic_and_instance_maps(
              semantic_maps,
              instance_maps,
              thing_ids,
              label_divisor=label_divisor,
              stuff_area_limit=stuff_area_limit,
              void_label=void_label))
      parsing_maps_gpu = parsing_maps_gpu.numpy()

    # Checks semantic maps are the same.
    semantic_maps_cpu = parsing_maps_cpu // label_divisor
    semantic_maps_gpu = parsing_maps_gpu // label_divisor
    np.testing.assert_array_equal(semantic_maps_cpu, semantic_maps_gpu)

    # Checks instance maps are the same, despite of label order.
    instance_maps_cpu = parsing_maps_cpu % label_divisor
    instance_maps_gpu = parsing_maps_gpu % label_divisor

    thing_labels_cpu = np.unique(semantic_maps_cpu[instance_maps_cpu > 0])
    for semantic_label in thing_labels_cpu:
      semantic_mask = semantic_maps_cpu == semantic_label
      instance_labels_cpu = np.unique(instance_maps_cpu[semantic_mask])
      instance_labels_gpu = np.unique(instance_maps_gpu[semantic_mask])

      self.assertEqual(len(instance_labels_cpu), len(instance_labels_gpu))

      # For each instance (cpu reference) of this semantic label, we check:
      # 1. Within this instance mask, GPU produces one and only one instance
      #    label.
      # 2. GPU results with the current semantic and instance label matches
      #    CPU instance mask.
      for instance_label in instance_labels_cpu:
        instance_mask_cpu = np.logical_and(
            instance_maps_cpu == instance_label, semantic_mask)
        instance_labels_gpu = set(instance_maps_gpu[instance_mask_cpu])
        self.assertLen(instance_labels_gpu, 1)

        instance_label_gpu = instance_labels_gpu.pop()
        # Here GPU must use the same semantic mask (given we have checked
        # semantic maps are the same).
        instance_mask_gpu = np.logical_and(
            instance_maps_gpu == instance_label_gpu, semantic_mask)
        np.testing.assert_array_equal(instance_mask_cpu, instance_mask_gpu)


if __name__ == '__main__':
  tf.test.main()
