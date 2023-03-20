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

"""Implementation of the metrics for monocular depth estimation.

The metrics for monocular depth estimation includes scale invariant logarithmic
error (SILog), relative squared error (SqErrorRel), relative absolute error
(AbsErrorRel), and depth inlier metric (DepthInlier) at 1.25.

Please see the papers for details:
[1] "Sparsity Invariant CNNs", Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe
  Franke, Thomas Brox, and Andreas Geiger. 3DV 2017.
[2] "Depth map prediction from a single image using a multi-scale deep network",
  David Eigen, Christian Puhrsch, and Rob Fergus. NeurIPS 2014.
"""

from typing import Dict, Text, Any
import tensorflow as tf


class DepthMetrics(tf.keras.metrics.Metric):
  """Metric class for the metrics for monocular depth estimation.

  The metrics for monocular depth estimation includes scale invariant
  logarithmic error (SILog), relative squared error (SqErrorRel), relative
  absolute error (AbsErrorRel), and depth inlier metric (DepthInlier) at 1.25.

  [1] "Sparsity Invariant CNNs", Jonas Uhrig, Nick Schneider, Lukas Schneider,
    Uwe Franke, Thomas Brox, and Andreas Geiger. 3DV 2017.
  [2] "Depth map prediction from a single image using a multi-scale deep
    network", David Eigen, Christian Puhrsch, and Rob Fergus. NeurIPS 2014.

  Stand-alone usage:

  depth_metric_obj = depth_metrics.DepthMetrics()
  depth_metric_obj.update_state(y_true_1, y_pred_1)
  depth_metric_obj.update_state(y_true_2, y_pred_2)
  ...
  result = depth_metric_obj.result()
  """

  def __init__(self, name: str = 'depth_metrics', **kwargs):
    """Initialization for the depth metrics."""
    super().__init__(name=name, **kwargs)
    self._silog = []
    self._sq_error_rel = []
    self._abs_error_rel = []
    self._depth_inlier = []

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """Accumulates the depth metrics statistics.

    Args:
      y_true: The ground truth depth label map.
      y_pred: The predicted depth map.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    valid_mask = y_true > 0
    y_true = tf.boolean_mask(y_true, valid_mask)
    y_pred = tf.boolean_mask(y_pred, valid_mask)
    # Computes SILog.
    err = tf.math.log(y_pred) - tf.math.log(y_true)
    silog = tf.sqrt(tf.reduce_mean(err**2) - tf.reduce_mean(err)**2) * 100
    self._silog.append(silog)
    # Computes SqErrorRel.
    sq_error_rel = tf.reduce_mean(((y_true - y_pred) / y_true)**2)
    self._sq_error_rel.append(sq_error_rel)
    # Computes AbsErrorRel.
    abs_error_rel = tf.reduce_mean(tf.abs(y_true - y_pred) / y_true)
    self._abs_error_rel.append(abs_error_rel)
    # Computes depth inlier.
    depth_inlier = tf.maximum(y_true / y_pred, y_pred / y_true) < 1.25
    depth_inlier = tf.reduce_mean(tf.cast(depth_inlier, dtype=tf.float32))
    self._depth_inlier.append(depth_inlier)

  def result(self) -> Dict[Text, Any]:
    """Computes the depth metrics."""
    silog = tf.convert_to_tensor(self._silog, dtype=tf.float32)
    sq_error_rel = tf.convert_to_tensor(self._sq_error_rel, dtype=tf.float32)
    abs_error_rel = tf.convert_to_tensor(self._abs_error_rel, dtype=tf.float32)
    depth_inlier = tf.convert_to_tensor(self._depth_inlier, dtype=tf.float32)
    metric_results = tf.stack(
        [silog, sq_error_rel, abs_error_rel, depth_inlier], axis=0)
    results = tf.reduce_mean(metric_results, axis=1)
    return results

  def reset_states(self):
    """Resets all stats that accumulates data."""
    self._silog = []
    self._sq_error_rel = []
    self._abs_error_rel = []
    self._depth_inlier = []
