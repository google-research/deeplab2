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

"""Utility script to perform net surgery on a model.

This script will perform net surgery on DeepLab models trained on a source
dataset and create a new checkpoint for the target dataset.
"""

from typing import Any, Dict, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import deeplab

FLAGS = flags.FLAGS

flags.DEFINE_string('source_dataset', 'cityscapes',
                    'Dataset name on which the model has been pretrained. '
                    'Supported datasets: `cityscapes`.')

flags.DEFINE_string('target_dataset', 'motchallenge_step',
                    'Dataset name for conversion. Supported datasets: '
                    '`motchallenge_step`.')

flags.DEFINE_string('input_config_path', None,
                    'Path to a config file that defines the DeepLab model and '
                    'the checkpoint path.')

flags.DEFINE_string('output_checkpoint_path', None,
                    'Output filename for the generated checkpoint file.')


_SUPPORTED_SOURCE_DATASETS = {'cityscapes'}
_SUPPORTED_TARGET_DATASETS = {'motchallenge_step'}

_CITYSCAPES_TO_MOTCHALLENGE_STEP = (
    1,  # sidewalk
    2,  # building
    8,  # vegetation
    10,  # sky
    11,  # pedestrian
    12,  # rider
    18,  # bicycle
)

_DATASET_TO_INFO = {
    'cityscapes': dataset.CITYSCAPES_PANOPTIC_INFORMATION,
    'motchallenge_step': dataset.MOTCHALLENGE_STEP_INFORMATION,
}
_INPUT_SIZE = (1025, 2049, 3)


def _load_model(
    config_path: Text,
    source_dataset: Text) -> Tuple[deeplab.DeepLab,
                                   config_pb2.ExperimentOptions]:
  """Load DeepLab model based on config and dataset."""
  options = config_pb2.ExperimentOptions()
  with tf.io.gfile.GFile(config_path) as f:
    text_format.Parse(f.read(), options)
  options.model_options.panoptic_deeplab.semantic_head.output_channels = (
      _DATASET_TO_INFO[source_dataset].num_classes)
  model = deeplab.DeepLab(options,
                          _DATASET_TO_INFO[source_dataset])
  return model, options


def _convert_bias(input_tensor: np.ndarray,
                  label_list: Tuple[int, ...]) -> np.ndarray:
  """Converts 1D tensor bias w.r.t. label list.

  We select the subsets from the input_tensor based on the label_list.

  We assume input_tensor has shape = [num_classes], where
  input_tensor is the bias weights trained on source dataset, and num_classes
  is the number of classes in source dataset.

  Args:
    input_tensor: A numpy array with ndim == 1.
    label_list: A tuple of labels used for net surgery.

  Returns:
    A numpy array with values modified.

  Raises:
    ValueError: input_tensor's ndim != 1.
  """
  if input_tensor.ndim != 1:
    raise ValueError('The bias tensor should have ndim == 1.')

  num_elements = len(label_list)
  output_tensor = np.zeros(num_elements, dtype=np.float32)
  for i, label in enumerate(label_list):
    output_tensor[i] = input_tensor[label]
  return output_tensor


def _convert_kernels(input_tensor: np.ndarray,
                     label_list: Tuple[int, ...]) -> np.ndarray:
  """Converts 4D tensor kernels w.r.t. label list.

  We select the subsets from the input_tensor based on the label_list.

  We assume input_tensor has shape = [h, w, input_dim, num_classes], where
  input_tensor is the kernel weights trained on source dataset, and num_classes
  is the number of classes in source dataset.

  Args:
    input_tensor: A numpy array with ndim == 4.
    label_list: A tuple of labels used for net surgery.

  Returns:
    A numpy array with values modified.

  Raises:
    ValueError: input_tensor's ndim != 4.
  """
  if input_tensor.ndim != 4:
    raise ValueError('The kernels tensor should have ndim == 4.')

  num_elements = len(label_list)
  kernel_height, kernel_width, input_dim, _ = input_tensor.shape
  output_tensor = np.zeros(
      (kernel_height, kernel_width, input_dim, num_elements), dtype=np.float32)
  for i, label in enumerate(label_list):
    output_tensor[:, :, :, i] = input_tensor[:, :, :, label]
  return output_tensor


def _restore_checkpoint(restore_dict: Dict[Any, Any],
                        options: config_pb2.ExperimentOptions
                        ) -> tf.train.Checkpoint:
  """Reads the provided dict items from the checkpoint specified in options.

  Args:
    restore_dict: A mapping of checkpoint item to location.
    options: A experiment configuration containing the checkpoint location.

  Returns:
    The loaded checkpoint.
  """
  ckpt = tf.train.Checkpoint(**restore_dict)
  if tf.io.gfile.isdir(options.model_options.initial_checkpoint):
    path = tf.train.latest_checkpoint(
        options.model_options.initial_checkpoint)
    status = ckpt.restore(path)
  else:
    status = ckpt.restore(options.model_options.initial_checkpoint)
  status.expect_partial().assert_existing_objects_matched()
  return ckpt


def main(_) -> None:
  if FLAGS.source_dataset not in _SUPPORTED_SOURCE_DATASETS:
    raise ValueError('Source dataset is not supported. Use --help to get list '
                     'of supported datasets.')
  if FLAGS.target_dataset not in _SUPPORTED_TARGET_DATASETS:
    raise ValueError('Target dataset is not supported. Use --help to get list '
                     'of supported datasets.')

  logging.info('Loading DeepLab model from config %s', FLAGS.input_config_path)
  source_model, options = _load_model(FLAGS.input_config_path,
                                      FLAGS.source_dataset)
  logging.info('Load pretrained checkpoint.')
  _restore_checkpoint(source_model.checkpoint_items, options)
  source_model(tf.keras.Input(_INPUT_SIZE), training=False)

  logging.info('Perform net surgery.')
  semantic_weights = (
      source_model._decoder._semantic_head.final_conv.get_weights())  # pylint: disable=protected-access

  if (FLAGS.source_dataset == 'cityscapes' and
      FLAGS.target_dataset == 'motchallenge_step'):
    # Kernels.
    semantic_weights[0] = _convert_kernels(semantic_weights[0],
                                           _CITYSCAPES_TO_MOTCHALLENGE_STEP)
    # Bias.
    semantic_weights[1] = _convert_bias(semantic_weights[1],
                                        _CITYSCAPES_TO_MOTCHALLENGE_STEP)

  logging.info('Load target model without last semantic layer.')
  target_model, _ = _load_model(FLAGS.input_config_path, FLAGS.target_dataset)
  restore_dict = target_model.checkpoint_items
  del restore_dict[common.CKPT_SEMANTIC_LAST_LAYER]

  ckpt = _restore_checkpoint(restore_dict, options)
  target_model(tf.keras.Input(_INPUT_SIZE), training=False)
  target_model._decoder._semantic_head.final_conv.set_weights(semantic_weights)  # pylint: disable=protected-access

  logging.info('Save checkpoint to output path: %s',
               FLAGS.output_checkpoint_path)
  ckpt = tf.train.Checkpoint(**target_model.checkpoint_items)
  ckpt.save(FLAGS.output_checkpoint_path)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['input_config_path', 'output_checkpoint_path'])
  app.run(main)
