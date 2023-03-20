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

"""Tests for matchers_ops."""

import numpy as np
from scipy import optimize
import tensorflow as tf

from deeplab2.model.loss import matchers_ops


class MatchersOpsTest(tf.test.TestCase):

  def hungarian_matching_tpu(self, cost_matrix):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    @tf.function
    def function():
      costs = tf.constant(cost_matrix, cost_matrix.dtype, cost_matrix.shape)
      return matchers_ops.hungarian_matching(costs)
    # Get the first replica output.
    return strategy.run(function).values[0].numpy()

  def testLinearSumAssignment(self):
    """Check a simple 2D test case of the Linear Sum Assignment problem.

    Ensures that the implementation of the matching algorithm is correct
    and functional on TPUs.
    """
    cost_matrix = np.array([[[4, 1, 3], [2, 0, 5], [3, 2, 2]]],
                           dtype=np.float32)
    adjacency_output = self.hungarian_matching_tpu(cost_matrix)

    correct_output = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=bool)
    self.assertAllEqual(adjacency_output[0], correct_output)

  def testBatchedLinearSumAssignment(self):
    """Check a batched case of the Linear Sum Assignment Problem.

    Ensures that a correct solution is found for all inputted problems within
    a batch.
    """
    cost_matrix = np.array([
        [[4, 1, 3], [2, 0, 5], [3, 2, 2]],
        [[1, 4, 3], [0, 2, 5], [2, 3, 2]],
        [[1, 3, 4], [0, 5, 2], [2, 2, 3]],
    ],
                           dtype=np.float32)

    adjacency_output = self.hungarian_matching_tpu(cost_matrix)

    # Hand solved correct output for the linear sum assignment problem
    correct_output = np.array([
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    ],
                              dtype=bool)
    self.assertAllClose(adjacency_output, correct_output)

  def testMaximumBipartiteMatching(self):
    """Check that the maximum bipartite match assigns the correct numbers."""
    adj_matrix = tf.cast([[
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]], tf.bool)  # pyformat: disable
    _, assignment = matchers_ops._maximum_bipartite_matching(adj_matrix)
    self.assertEqual(np.sum(assignment), 5)

  def testAssignmentMatchesScipy(self):
    """Check that the Linear Sum Assignment matches the Scipy implementation."""
    batch_size, num_elems = 2, 25
    weights = tf.random.uniform((batch_size, num_elems, num_elems),
                                minval=0.,
                                maxval=1.)
    assignment = matchers_ops.hungarian_matching(weights)
    actual_weights = weights.numpy()
    actual_assignment = assignment.numpy()

    for idx in range(batch_size):
      _, scipy_assignment = optimize.linear_sum_assignment(actual_weights[idx])
      hungarian_assignment = np.where(actual_assignment[idx])[1]

      self.assertAllEqual(hungarian_assignment, scipy_assignment)

  def testAssignmentRunsOnTPU(self):
    """Check that a batch of assignments matches Scipy."""
    batch_size, num_elems = 4, 100
    cost_matrix = np.random.rand(batch_size, num_elems, num_elems)

    actual_assignment = self.hungarian_matching_tpu(cost_matrix)

    for idx in range(batch_size):
      _, scipy_assignment = optimize.linear_sum_assignment(cost_matrix[idx])
      hungarian_assignment = np.where(actual_assignment[idx])[1]
      self.assertAllEqual(hungarian_assignment, scipy_assignment)

  def testLargeBatch(self):
    """Check large-batch performance of Hungarian matcher.

    Useful for testing efficiency of the proposed solution and regression
    testing. Current solution is thought to be quadratic in nature, yielding
    significant slowdowns when the number of queries is increased.
    """
    batch_size, num_elems = 64, 100
    cost_matrix = np.abs(
        np.random.normal(size=(batch_size, num_elems, num_elems)))

    _ = self.hungarian_matching_tpu(cost_matrix)


if __name__ == '__main__':
  tf.test.main()
