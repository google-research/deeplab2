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

"""Tests for post_processor_builder.py."""

import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model.post_processor import post_processor_builder


class EvaluatorTest(tf.test.TestCase):

  def test_evaluates_panoptic_deeplab_model(self):
    experiment_options_textproto = """
      experiment_name: "evaluation_test"
      eval_dataset_options {
        dataset: "cityscapes_panoptic"
        file_pattern: "EMPTY"
        batch_size: 1
        crop_size: 1025
        crop_size: 2049
        # Skip resizing.
        min_resize_value: 0
        max_resize_value: 0
      }
      evaluator_options {
        continuous_eval_timeout: -1
        stuff_area_limit: 2048
        center_score_threshold: 0.1
        nms_kernel: 13
        save_predictions: true
        save_raw_predictions: false
      }
    """
    config = text_format.Parse(experiment_options_textproto,
                               config_pb2.ExperimentOptions())
    config.model_options.panoptic_deeplab.instance.enable = True
    post_processor = post_processor_builder.get_post_processor(
        config, dataset.CITYSCAPES_PANOPTIC_INFORMATION)

    result_dict = {
        common.PRED_SEMANTIC_PROBS_KEY:
            tf.zeros([1, 1025, 2049, 19], dtype=tf.float32),
        common.PRED_CENTER_HEATMAP_KEY:
            tf.zeros([1, 1025, 2049, 1], dtype=tf.float32),
        common.PRED_OFFSET_MAP_KEY:
            tf.zeros([1, 1025, 2049, 2], dtype=tf.float32)
    }
    processed_dict = post_processor(result_dict)
    expected_keys = {
        common.PRED_PANOPTIC_KEY,
        common.PRED_SEMANTIC_KEY,
        common.PRED_INSTANCE_KEY,
        common.PRED_INSTANCE_CENTER_KEY,
        common.PRED_INSTANCE_SCORES_KEY
    }
    self.assertCountEqual(processed_dict.keys(), expected_keys)


if __name__ == '__main__':
  tf.test.main()
