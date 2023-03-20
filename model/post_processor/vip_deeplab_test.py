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

"""Test for vip_deeplab.py."""
import numpy as np
import tensorflow as tf

from deeplab2.model.post_processor import vip_deeplab


class PostProcessingTest(tf.test.TestCase):

  def test_stitch_video_panoptic_prediction(self):
    concat_semantic = np.array(
        [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 2, 0], [2, 2, 3, 3]]],
        dtype=np.int32)
    concat_instance = np.array(
        [[[1, 1, 2, 2], [1, 0, 0, 2], [1, 1, 1, 2], [2, 2, 1, 1]]],
        dtype=np.int32)
    next_semantic = np.array(
        [[[0, 1, 1, 0], [0, 1, 1, 0], [0, 2, 2, 0], [2, 2, 3, 3]]],
        dtype=np.int32)
    next_instance = np.array(
        [[[2, 0, 0, 1], [2, 0, 0, 1], [2, 4, 4, 1], [5, 5, 3, 3]]],
        dtype=np.int32)
    label_divisor = 1000
    concat_panoptic = concat_semantic * label_divisor + concat_instance
    next_panoptic = next_semantic * label_divisor + next_instance
    new_panoptic = vip_deeplab.stitch_video_panoptic_prediction(
        concat_panoptic, next_panoptic, label_divisor)
    # The expected instance is manually computed. It should receive the IDs
    # propagated from concat_instance by IoU matching between concat_panoptic
    # and next_panoptic.
    expected_semantic = next_semantic
    expected_instance = np.array(
        [[[1, 0, 0, 2], [1, 0, 0, 2], [1, 1, 1, 2], [2, 2, 1, 1]]],
        dtype=np.int32)
    expected_panoptic = expected_semantic * label_divisor + expected_instance
    np.testing.assert_array_equal(expected_panoptic, new_panoptic)

  def test_tf_video_panoptic_prediction_stitcher(self):
    concat_semantic = np.array(
        [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 2, 0], [2, 2, 3, 3]]],
        dtype=np.int32)
    concat_instance = np.array(
        [[[1, 1, 2, 2], [1, 0, 0, 2], [1, 1, 1, 2], [2, 2, 1, 1]]],
        dtype=np.int32)
    next_semantic = np.array(
        [[[0, 1, 1, 0], [0, 1, 1, 0], [0, 2, 2, 0], [2, 2, 3, 3]]],
        dtype=np.int32)
    next_instance = np.array(
        [[[2, 0, 0, 1], [2, 0, 0, 1], [2, 4, 4, 1], [5, 5, 3, 3]]],
        dtype=np.int32)
    label_divisor = 1000
    concat_panoptic = concat_semantic * label_divisor + concat_instance
    next_panoptic = next_semantic * label_divisor + next_instance
    stitcher = vip_deeplab.VideoPanopticPredictionStitcher(label_divisor)
    new_panoptic = stitcher(
        tf.convert_to_tensor(concat_panoptic),
        tf.convert_to_tensor(next_panoptic)).numpy()
    # The expected instance is manually computed. It should receive the IDs
    # propagated from concat_instance by IoU matching between concat_panoptic
    # and next_panoptic.
    expected_semantic = next_semantic
    expected_instance = np.array(
        [[[1, 0, 0, 2], [1, 0, 0, 2], [1, 1, 1, 2], [2, 2, 1, 1]]],
        dtype=np.int32)
    expected_panoptic = expected_semantic * label_divisor + expected_instance
    np.testing.assert_array_equal(expected_panoptic, new_panoptic)


if __name__ == '__main__':
  tf.test.main()
