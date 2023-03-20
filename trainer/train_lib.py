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

"""This file contains code to create run an experiment."""
import functools
import os
from typing import Text, Optional, Sequence

from absl import logging
import orbit
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import deeplab
from deeplab2.model import kmax_deeplab
from deeplab2.model.loss import loss_builder
from deeplab2.trainer import distribution_utils
from deeplab2.trainer import evaluator as evaluator_lib
from deeplab2.trainer import runner_utils
from deeplab2.trainer import trainer as trainer_lib
from deeplab2.video import motion_deeplab
from deeplab2.video import vip_deeplab

_INSTANCE_LAYER_NAMES = (common.CKPT_MOTION_REGRESSION_HEAD_LAST_LAYER,
                         common.CKPT_INSTANCE_REGRESSION_HEAD_LAST_LAYER,
                         common.CKPT_INSTANCE_CENTER_HEAD_LAST_LAYER)
# For Motion-Deeplab, 6 channels are used as input (2x RGB) during inference.
# Its 7th input channel is obtained by the predicted center heatmap of
# previous frame.
_TWO_FRAME_MOTION_DEEPLAB_INPUT_CHANNELS = 6
# All other networks use 3 channels as input (RGB).
_SINGLE_FRAME_INPUT_CHANNELS = 3


def create_deeplab_model(
    config: config_pb2.ExperimentOptions,
    dataset_descriptor: dataset.DatasetDescriptor) -> tf.keras.Model:
  """Creates DeepLab model based on config."""
  if config.model_options.WhichOneof('meta_architecture') == 'motion_deeplab':
    return motion_deeplab.MotionDeepLab(config, dataset_descriptor)
  elif config.model_options.WhichOneof('meta_architecture') == 'vip_deeplab':
    return vip_deeplab.ViPDeepLab(config, dataset_descriptor)
  elif ('kmax_' in config.model_options.backbone.name and
        config.model_options.WhichOneof('meta_architecture') == 'max_deeplab'):
    return kmax_deeplab.KMaXDeepLab(config, dataset_descriptor)
  else:
    return deeplab.DeepLab(config, dataset_descriptor)


def build_deeplab_model(deeplab_model: tf.keras.Model,
                        crop_size: Sequence[int],
                        batch_size: Optional[int] = None):
  """Builds DeepLab model with input crop size."""
  if isinstance(deeplab_model, motion_deeplab.MotionDeepLab) or isinstance(
      deeplab_model, vip_deeplab.ViPDeepLab):
    # Motion-DeepLab and ViP-DeepLab use the input differently despite that
    # the input_shape is the same: Motion-DeepLab uses two frames as one input,
    # while ViP-DeepLab splits the two frames first and passes them individually
    # to the backbone encoder.
    input_shape = list(crop_size) + [_TWO_FRAME_MOTION_DEEPLAB_INPUT_CHANNELS]
    deeplab_model(
        tf.keras.Input(input_shape, batch_size=batch_size), training=False)
  else:
    input_shape = list(crop_size) + [_SINGLE_FRAME_INPUT_CHANNELS]
    deeplab_model(
        tf.keras.Input(input_shape, batch_size=batch_size), training=False)
  return input_shape


def run_experiment(mode: Text, config: config_pb2.ExperimentOptions,
                   model_dir: Text, tpu: Optional[Text], num_gpus: int):
  """Runs an experiment.

  Args:
    mode: A string specifying the mode of the experiment. Supported are `train`,
      `train_and_eval`, `eval` and `continuous_eval`.
    config: A config_pb2.ExperimentOptions configuration.
    model_dir: A path to store all checkpoints and other experimental artifacts.
    tpu: The name or address of the tpu to connect to, if any.
    num_gpus: An integer specifying the number of GPUs to use. If mode contains
      `eval`, num_gpus must be less or equal to 1.

  Raises:
    ValueError: If mode is none of `train`, `train_and_eval`, `eval`, or
      `continuous_eval`.
    ValueError: If mode is `train_and_eval`, but different dataset_names are
      specified for training and evaluation. This error could be relaxed for
      applications like domain transferring learning (e.g., synthetic to real
      datasets), which has not been fully tested yet.
    ValueError: If mode includes `eval` and num_gpus > 1. Currently, evaluation
      is not supported on more than a single GPU.
  """
  strategy = distribution_utils.create_strategy(tpu, num_gpus)
  logging.info('Using strategy %s with %d replicas', type(strategy),
               strategy.num_replicas_in_sync)

  if 'eval' in mode:
    dataset_name = config.eval_dataset_options.dataset
    if (mode == 'train_and_eval' and
        dataset_name != config.train_dataset_options.dataset):
      logging.warning('Using difference dataset_names in train_and_eval mode.'
                      ' Please confirm this is the desired behavior.')
    if num_gpus > 1:
      raise ValueError(
          'Using more than one GPU for evaluation is not supported.')
  else:
    dataset_name = config.train_dataset_options.dataset

  num_classes = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].num_classes
  ignore_label = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_label
  ignore_depth = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_depth
  class_has_instances_list = (
      dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].class_has_instances_list)

  trainer = None
  evaluator = None
  with strategy.scope():
    deeplab_model = create_deeplab_model(
        config, dataset.MAP_NAME_TO_DATASET_INFO[dataset_name])
    losses = loss_builder.DeepLabFamilyLoss(
        loss_options=config.trainer_options.loss_options,
        deeplab_options=config.model_options,
        num_classes=num_classes,
        ignore_label=ignore_label,
        ignore_depth=ignore_depth,
        thing_class_ids=class_has_instances_list,
        auxiliary_output_number=deeplab_model.auxiliary_output_number)
    losses_eval = loss_builder.DeepLabFamilyLoss(
        loss_options=config.trainer_options.loss_options,
        deeplab_options=config.model_options,
        num_classes=num_classes,
        ignore_label=ignore_label,
        ignore_depth=ignore_depth,
        thing_class_ids=class_has_instances_list,
        auxiliary_output_number=0)
    global_step = orbit.utils.create_global_step()
    if 'train' in mode:
      trainer = trainer_lib.Trainer(config, deeplab_model, losses, global_step)
    if 'eval' in mode:
      evaluator = evaluator_lib.Evaluator(config, deeplab_model, losses_eval,
                                          global_step, model_dir)

  checkpoint_dict = dict(global_step=global_step)
  checkpoint_dict.update(deeplab_model.checkpoint_items)
  if trainer is not None:
    checkpoint_dict['optimizer'] = trainer.optimizer
    if trainer.backbone_optimizer is not None:
      checkpoint_dict['backbone_optimizer'] = trainer.backbone_optimizer
  checkpoint = tf.train.Checkpoint(**checkpoint_dict)

  # Define items to load from initial checkpoint.
  init_dict = deeplab_model.checkpoint_items
  if (not config.model_options
      .restore_semantic_last_layer_from_initial_checkpoint):
    del init_dict[common.CKPT_SEMANTIC_LAST_LAYER]
  if (not config.model_options
      .restore_instance_last_layer_from_initial_checkpoint):
    for layer_name in _INSTANCE_LAYER_NAMES:
      if layer_name in init_dict:
        del init_dict[layer_name]
  init_fn = functools.partial(runner_utils.maybe_load_checkpoint,
                              config.model_options.initial_checkpoint,
                              init_dict)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=config.trainer_options.num_checkpoints_to_keep,
      step_counter=global_step,
      checkpoint_interval=config.trainer_options.save_checkpoints_steps,
      init_fn=init_fn)

  controller = orbit.Controller(
      strategy=strategy,
      trainer=trainer,
      evaluator=evaluator,
      global_step=global_step,
      steps_per_loop=config.trainer_options.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_interval=config.trainer_options.save_summaries_steps,
      summary_dir=os.path.join(model_dir, 'train'),
      eval_summary_dir=os.path.join(model_dir, 'eval'))

  with strategy.scope():
    # Save initial checkpoint.
    if 'train' in mode:
      crop_size = list(config.train_dataset_options.crop_size)
      # Build model before saving.
      build_deeplab_model(deeplab_model, crop_size)
      controller.save_checkpoint()
    if mode == 'train':
      controller.train(
          steps=config.trainer_options.solver_options.training_number_of_steps)
    elif mode == 'train_and_eval':
      # Interleave training and evaluation.
      controller.train_and_evaluate(
          train_steps=(
              config.trainer_options.solver_options.training_number_of_steps),
          eval_steps=config.evaluator_options.eval_steps,
          eval_interval=config.evaluator_options.eval_interval)
    elif mode == 'eval':
      controller.evaluate(steps=config.evaluator_options.eval_steps)
    elif mode == 'continuous_eval':
      # Monitor the checkpoint directory for new checkpoints to evaluate.
      timeout = config.evaluator_options.continuous_eval_timeout
      if timeout == -1:
        # Wait forever
        timeout = None
      controller.evaluate_continuously(
          steps=config.evaluator_options.eval_steps, timeout=timeout)
    else:
      raise ValueError('Mode %s is not a valid mode.' % mode)
