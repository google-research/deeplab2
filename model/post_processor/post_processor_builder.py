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

"""This file contains a post-processor builder used in the DeepLab model."""

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import utils
from deeplab2.model.post_processor import max_deeplab
from deeplab2.model.post_processor import panoptic_deeplab


def get_post_processor(
    config: config_pb2.ExperimentOptions,
    dataset_descriptor: dataset.DatasetDescriptor) -> tf.keras.layers.Layer:
  """Initializes a DeepLab post-processor.

  Args:
    config: A config_pb2.ExperimentOptions configuration.
    dataset_descriptor: A dataset.DatasetDescriptor.

  Returns:
    PostProcessor: A post-processor depending on the configuration.
  """
  supported_tasks = utils.get_supported_tasks(config)
  if config.model_options.WhichOneof('meta_architecture') == 'max_deeplab':
    return max_deeplab.PostProcessor(config, dataset_descriptor)
  if common.TASK_PANOPTIC_SEGMENTATION in supported_tasks:
    return panoptic_deeplab.PostProcessor(config, dataset_descriptor)
  return panoptic_deeplab.SemanticOnlyPostProcessor()
