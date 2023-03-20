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

"""Tests for depth metrics."""
import numpy as np
import tensorflow as tf

from deeplab2.evaluation import depth_metrics


class DepthMetricsTest(tf.test.TestCase):

  def test_depth_metrics_on_single_image(self):
    gt = np.array([[5.44108091, 53.30197697, 61.06181767, 14.36723114],
                   [0, 39.68081126, 58.77974067, 0],
                   [40.57883826, 22.15134852, 31.46813478, 13.52603324]])
    pred = np.array([[4.87694111, 50.09085582, 55.74533641, 10.13579195],
                     [13.76178147, 41.62431592, 56.97362032, 81.48369608],
                     [43.12005689, 15.54622258, 24.1993478, 12.14451783]])
    depth_obj = depth_metrics.DepthMetrics()
    depth_obj.update_state(gt, pred)
    result = depth_obj.result().numpy()
    # The following numbers are manually computed.
    self.assertAlmostEqual(result[0], 14.154233, places=4)
    self.assertAlmostEqual(result[1], 0.0268667, places=4)
    self.assertAlmostEqual(result[2], 0.13191505, places=4)
    self.assertAlmostEqual(result[3], 0.7, places=4)

  def test_depth_metrics_on_multiple_images(self):
    depth_obj = depth_metrics.DepthMetrics()
    gt_1 = np.array([[5.44108091, 53.30197697, 61.06181767, 14.36723114],
                     [0, 39.68081126, 58.77974067, 0],
                     [40.57883826, 22.15134852, 31.46813478, 13.52603324]])
    pred_1 = np.array([[4.87694111, 50.09085582, 55.74533641, 10.13579195],
                       [13.76178147, 41.62431592, 56.97362032, 81.48369608],
                       [43.12005689, 15.54622258, 24.1993478, 12.14451783]])
    depth_obj.update_state(gt_1, pred_1)
    gt_2 = np.array(
        [[79.56192404, 25.68145225, 0, 39.88486608, 68.91602466],
         [79.53460057, 2.55741031, 36.05057241, 68.04747416, 3.7783227],
         [0, 0, 72.47336778, 59.02611644, 66.07499008],
         [25.88578395, 58.2202574, 27.39066477, 29.83094038, 37.99239669]])
    pred_2 = np.array(
        [[83.80952145, 27.23367361, 72.52687468, 35.28400183, 72.41126444],
         [77.62373864, 0.87004049, 32.1619225, 66.91361903, 2.60688436],
         [15.30294603, 9.76419241, 68.61650198, 57.14559324, 66.88452603],
         [24.54818109, 61.60855251, 31.50312052, 26.02325866, 36.4019569]])
    depth_obj.update_state(gt_2, pred_2)
    gt_3 = np.array([[50.80100791, 0.41130084, 58.85031668],
                     [29.44932853, 23.48806627, 30.17890056]])
    pred_3 = np.array([[49.66563966, 0.62070026, 58.84231026],
                       [32.26735775, 28.07405648, 33.7131882]])
    depth_obj.update_state(gt_3, pred_3)
    result = depth_obj.result().numpy()
    # The following numbers are manually computed.
    self.assertAlmostEqual(result[0], 18.442057, places=4)
    self.assertAlmostEqual(result[1], 0.0388692, places=4)
    self.assertAlmostEqual(result[2], 0.13392223, places=4)
    self.assertAlmostEqual(result[3], 0.8052287, places=4)


if __name__ == '__main__':
  tf.test.main()
