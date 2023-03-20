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

"""This file contains code to run a model."""

import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.trainer import train_lib

flags.DEFINE_enum(
    'mode',
    default=None,
    enum_values=['train', 'eval', 'train_and_eval', 'continuous_eval'],
    help='Mode to run: `train`, `eval`, `train_and_eval`, `continuous_eval`.')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help='The base directory where the model and training/evaluation summaries'
    'are stored. The path will be combined with the `experiment_name` defined '
    'in the config file to create a folder under which all files are stored.')

flags.DEFINE_string(
    'config_file',
    default=None,
    help='Proto file which specifies the experiment configuration. The proto '
    'definition of ExperimentOptions is specified in config.proto.')

flags.DEFINE_string(
    'master',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_integer(
    'num_gpus',
    default=0,
    help='The number of GPUs to use for. If `master` flag is not set, this'
    'parameter specifies whether GPUs should be used and how many of them '
    '(default: 0).')

FLAGS = flags.FLAGS


def main(_):
  logging.info('Reading the config file.')
  with tf.io.gfile.GFile(FLAGS.config_file, 'r') as proto_file:
    config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())

  logging.info('Starting the experiment.')
  combined_model_dir = os.path.join(FLAGS.model_dir, config.experiment_name)
  train_lib.run_experiment(FLAGS.mode, config, combined_model_dir, FLAGS.master,
                           FLAGS.num_gpus)


if __name__ == '__main__':
  app.run(main)
