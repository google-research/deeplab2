# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
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
from deeplab2.trainer import runner_utils
from deeplab2.video import motion_deeplab


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self,
               initial_learning_rate,
               decay_schedule_fn,
               warmup_steps,
               name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements linear warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = self.initial_learning_rate * warmup_percent_done
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step),
          name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'name': self.name
    }


def _create_optimizer(
    solver_config: config_pb2.SolverOptions) -> tf.keras.optimizers.Optimizer:
  """Creates an Optimizer based on the configuration.

  Args:
    solver_config: A trainer_pb2.SolverOptions configuration.

  Returns:
    A tf.keras.optimizer.Optimizer.

  Raises:
    ValueError: An error occurs when the desired optimizer or learning rate
      scheduler is not supported.
  """
  if solver_config.learning_policy == 'poly':
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=solver_config.base_learning_rate,
        decay_steps=solver_config.training_number_of_steps,
        end_learning_rate=solver_config.poly_end_learning_rate,
        power=solver_config.poly_learning_power,
        cycle=False)
  elif solver_config.learning_policy == 'cosine':
    lr_scheduler = tf.keras.experimental.CosineDecay(
        initial_learning_rate=solver_config.base_learning_rate,
        decay_steps=solver_config.training_number_of_steps,
        alpha=0.0)
  else:
    raise ValueError('Learning rate policy %s is not supported.' %
                     solver_config.learning_policy)

  if solver_config.warmup_steps:
    lr_scheduler = WarmUp(
        initial_learning_rate=solver_config.base_learning_rate,
        decay_schedule_fn=lr_scheduler,
        warmup_steps=solver_config.warmup_steps,
        name='linear_warmup')

  if solver_config.optimizer == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  elif solver_config.optimizer == 'sgd':
    # We use momentum = 0.9, the most frequently used case.
    return tf.keras.optimizers.SGD(learning_rate=lr_scheduler,
                                   momentum=0.9)

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
    self._has_instance_loss = (
        config.model_options.panoptic_deeplab.instance.enable)
    train_dataset = runner_utils.create_dataset(
        config.train_dataset_options,
        is_training=True,
        only_semantic_annotations=not self._has_instance_loss)
    train_dataset = orbit.utils.make_distributed_dataset(
        self.strategy, train_dataset)
    super(Trainer, self).__init__(train_dataset)

    self._config = config
    self._model = model
    self._loss = loss

    self._optimizer = _create_optimizer(config.trainer_options.solver_options)
    self._global_step = global_step
    self._use_gradient_clipping = (
        config.trainer_options.solver_options.use_gradient_clipping
        )
    self._clip_gradient_norm = (
        config.trainer_options.solver_options.clip_gradient_norm
        )
    self._has_motion_loss = isinstance(model, motion_deeplab.MotionDeepLab)

    self._train_total_loss_metric = tf.keras.metrics.Mean(
        'train_total_loss', dtype=tf.float32)
    self._train_semantic_loss_metric = tf.keras.metrics.Mean(
        'train_semantic_loss', dtype=tf.float32)

    if self._has_instance_loss:
      self._train_center_loss_metric = tf.keras.metrics.Mean(
          'train_center_loss', dtype=tf.float32)
      self._train_regression_loss_metric = tf.keras.metrics.Mean(
          'train_regression_loss', dtype=tf.float32)

    self._train_metric_list = [
        self._train_total_loss_metric, self._train_semantic_loss_metric
    ]
    if self._has_instance_loss:
      self._train_metric_list.append(self._train_center_loss_metric)
      self._train_metric_list.append(self._train_regression_loss_metric)
    if self._has_motion_loss:
      self._train_motion_loss_metric = tf.keras.metrics.Mean(
          'train_motion_loss', dtype=tf.float32)
      self._train_metric_list.append(self._train_motion_loss_metric)

  def train_loop_begin(self):
    """Called once at the beginning of the training loop.

    This method is called before dataset iterators creation.
    """
    for metric in self._train_metric_list:
      metric.reset_states()

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
      losses = tf.reduce_mean(self._loss(inputs, outputs), axis=0)
      loss_list = [losses[0], losses[1], losses[2], losses[3]]
      losses = tf.reduce_sum(losses)
      loss_list = [losses] + loss_list
      scaled_loss = losses / self.strategy.num_replicas_in_sync

    training_vars = self._model.trainable_variables
    gradients = tape.gradient(scaled_loss, training_vars)

    # Apply gradient clipping.
    if self._clip_gradient_norm > 0.0 and self._use_gradient_clipping:
      gradients, _ = tf.clip_by_global_norm(gradients, self._clip_gradient_norm)
    self._optimizer.apply_gradients(list(zip(gradients, training_vars)))

    for i, loss_metric in enumerate(self._train_metric_list):
      loss_metric.update_state(loss_list[i])

  def train_loop_end(self) -> Dict[Text, tf.Tensor]:
    """Called at the end of the training loop.

    The value returned from this function will be returned as-is from the
    train() method.

    Returns:
      A dictionary of `Tensors`, which will be written to logs and as
      TensorBoard summaries.
    """
    train_logs = {}
    for loss_metric in self._train_metric_list:
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
  def strategy(self):
    return self._strategy

  @property
  def global_step(self):
    return self._global_step

  @property
  def model(self):
    return self._model
