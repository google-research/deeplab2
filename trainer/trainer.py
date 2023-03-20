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

"""This file contains code to create a Trainer for training and validation."""

from typing import Dict, Any, Text
import orbit
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model import utils
from deeplab2.trainer import runner_utils
from deeplab2.trainer import trainer_utils


def _create_optimizer(
    solver_config: config_pb2.SolverOptions,
    learning_rate_multiplier: float = 1.0) -> tf.keras.optimizers.Optimizer:
  """Creates an Optimizer based on the configuration.

  Args:
    solver_config: A trainer_pb2.SolverOptions configuration.
    learning_rate_multiplier: A float, the learning rate multiplier applied on
      top of the base learning rate. Default to 1.0.

  Returns:
    A tf.keras.optimizer.Optimizer.

  Raises:
    ValueError: An error occurs when the desired optimizer or learning rate
      scheduler is not supported.
  """
  learning_rate = (solver_config.base_learning_rate * learning_rate_multiplier)
  if solver_config.learning_policy == 'poly':
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=solver_config.training_number_of_steps,
        end_learning_rate=solver_config.poly_end_learning_rate,
        power=solver_config.poly_learning_power,
        cycle=False)
  elif solver_config.learning_policy == 'cosine':
    lr_scheduler = tf.keras.experimental.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=solver_config.training_number_of_steps,
        alpha=0.0)
  else:
    raise ValueError('Learning rate policy %s is not supported.' %
                     solver_config.learning_policy)

  if solver_config.warmup_steps:
    lr_scheduler = trainer_utils.WarmUp(
        initial_learning_rate=learning_rate,
        decay_schedule_fn=lr_scheduler,
        warmup_steps=solver_config.warmup_steps,
        name='linear_warmup')

  if solver_config.optimizer == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  elif solver_config.optimizer == 'sgd':
    # We use momentum = 0.9, the most frequently used case.
    return tf.keras.optimizers.SGD(learning_rate=lr_scheduler,
                                   momentum=0.9)
  elif solver_config.optimizer == 'adamw':
    return trainer_utils.AdamWeightDecay(
        learning_rate=lr_scheduler,
        weight_decay_rate=solver_config.adamw_weight_decay,
        # Weight decay is only applied to convolution/linear kernels/weights.
        include_in_weight_decay=[r'.*(kernel|weight):0$'],
        exclude_from_weight_decay=[r'.*$'],
        gradient_clip_norm=0.0)

  raise ValueError('Optimizer %s is not supported.' % solver_config.optimizer)


class Trainer(orbit.StandardTrainer):
  """Implements a Trainer for training DeepLab models."""

  def __init__(self, config: config_pb2.ExperimentOptions,
               model: tf.keras.Model, loss: tf.keras.losses.Loss,
               global_step: tf.Variable):
    """Initializes the trainer.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      model: A tf.keras.Model.
      loss: A tf.keras.losses.Loss.
      global_step: A tf.Variable that records the global training step.
    """
    self._strategy = tf.distribute.get_strategy()

    support_panoptic = (common.TASK_PANOPTIC_SEGMENTATION in
                        utils.get_supported_tasks(config))
    train_dataset = runner_utils.create_dataset(
        config.train_dataset_options,
        is_training=True,
        only_semantic_annotations=not support_panoptic)
    train_dataset = orbit.utils.make_distributed_dataset(
        self.strategy, train_dataset)
    super(Trainer, self).__init__(train_dataset)

    self._config = config
    self._model = model
    self._loss = loss

    solver_options = config.trainer_options.solver_options
    self._optimizer = _create_optimizer(solver_options)
    self._backbone_optimizer = None
    if solver_options.HasField('backbone_learning_rate_multiplier'):
      self._backbone_optimizer = _create_optimizer(
          solver_options, learning_rate_multiplier=(
              solver_options.backbone_learning_rate_multiplier))

    self._global_step = global_step
    self._use_gradient_clipping = solver_options.use_gradient_clipping
    self._clip_gradient_norm = solver_options.clip_gradient_norm

    self._train_loss_metric_dict = runner_utils.create_loss_metric_dict(
        loss.get_loss_names(), prefix='train_')

  def train_loop_begin(self):
    """Called once at the beginning of the training loop.

    This method is called before dataset iterators creation.
    """
    for metric in self._train_loss_metric_dict.values():
      metric.reset_states()

  def _apply_gradients_to_optimizers(self, gradients_and_variables):
    """Applies gradients to their optimizers.

    This function divides all trainable variables (and their gradients) into
    two groups. One group contains backbone variables that have been pretrained,
    e.g., on ImageNet classification. The other group contains all other
    variables that are added specifically for the dense prediction task, e.g.,
    panoptic segmentation. Then, we apply two optimizers, optionally with two
    learning rates, to the variables and gradients.

    Args:
      gradients_and_variables: A list of tuple of (gradient, variable) tensors.
    """
    if self._backbone_optimizer is None:
      self._optimizer.apply_gradients(gradients_and_variables)
    else:
      optimizer_inputs = []
      backbone_optimizer_inputs = []

      encoder = self._model.checkpoint_items['encoder']
      encoder_variable_names = [x.name for x in encoder.trainable_variables]
      encoder_name = self._config.model_options.backbone.name

      for gradient, variable in gradients_and_variables:
        if runner_utils.check_if_variable_in_backbone(variable, encoder_name,
                                                      encoder_variable_names):
          backbone_optimizer_inputs.append((gradient, variable))
        else:
          optimizer_inputs.append((gradient, variable))
      self._optimizer.apply_gradients(optimizer_inputs)
      self._backbone_optimizer.apply_gradients(backbone_optimizer_inputs)

  def train_step(self, iterator):
    """Implements one step of training.

    Runs one step of evaluation with respect to the chosen strategy. In case of
    a distributed strategy, the replica results are gathered and returned.

    Note that all operations within `_train_step` are tf.function compatible, as
    they will be traced with tf.function. Any other/numpy operations are put in
    `train_loop_begin` or `train_loop_end` functions.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.
    """

    def step_fn(inputs):
      self._train_step(inputs)
      self._global_step.assign_add(1)

    self._strategy.run(step_fn, args=(next(iterator),))

  def _train_step(self, inputs: Dict[Text, Any]):
    """Performs a forward and backward pass.

    Args:
      inputs: A dictionary to be consumed by the model.
    """
    with tf.GradientTape() as tape:
      outputs = self._model(inputs[common.IMAGE], training=True)
      # Get the average per-batch loss and scale it down by the number of
      # replicas. This ensures that we don't end up multiplying our loss by the
      # number of workers - gradients are summed, not averaged, across replicas
      # during the apply_gradients call.
      loss_dict = self._loss(inputs, outputs)
      # Average over the batch.
      average_loss_dict = {}
      for name, loss in loss_dict.items():
        averaged_loss = tf.reduce_mean(loss)
        average_loss_dict[name] = tf.where(tf.math.is_nan(averaged_loss),
                                           0.0, averaged_loss)

      total_loss = average_loss_dict[common.TOTAL_LOSS]
      scaled_loss = total_loss / self.strategy.num_replicas_in_sync

    training_vars = self._model.trainable_variables
    gradients = tape.gradient(scaled_loss, training_vars)

    # Apply gradient clipping.
    if self._clip_gradient_norm > 0.0 and self._use_gradient_clipping:
      gradients, _ = tf.clip_by_global_norm(gradients, self._clip_gradient_norm)

    self._apply_gradients_to_optimizers(list(zip(gradients, training_vars)))

    for name, value in average_loss_dict.items():
      self._train_loss_metric_dict[name].update_state(value)

  def train_loop_end(self) -> Dict[Text, tf.Tensor]:
    """Called at the end of the training loop.

    The value returned from this function will be returned as-is from the
    train() method.

    Returns:
      A dictionary of `Tensors`, which will be written to logs and as
      TensorBoard summaries.
    """
    train_logs = {}
    for loss_metric in self._train_loss_metric_dict.values():
      train_logs['losses/' + loss_metric.name] = loss_metric.result()

    if callable(self._optimizer.learning_rate):
      train_logs['learning_rate'] = self._optimizer.learning_rate(
          self._global_step)
    else:
      train_logs['learning_rate'] = self._optimizer.learning_rate
    return train_logs

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def backbone_optimizer(self):
    return self._backbone_optimizer

  @property
  def strategy(self):
    return self._strategy

  @property
  def global_step(self):
    return self._global_step

  @property
  def model(self):
    return self._model
