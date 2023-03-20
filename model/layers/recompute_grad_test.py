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

"""Tests for recompute_grad.

This file is based on the recompute_grad_test.py in the etcmodel codebase [1]:
https://github.com/google-research/google-research/blob/ae9d07f22d31b36069bb8321e9d015e46dd8e8bb/etcmodel/layers/recompute_grad_test.py

[1] ETC: Encoding Long and Structured Inputs in Transformers, EMNLP 2020.
      Joshua Ainslie, Santiago Ontanon, Chris Alberti, Vaclav Cvicek, Zachary
      Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, Li Yang.
"""
from typing import Sequence
import tensorflow as tf
from deeplab2.model import test_utils
from deeplab2.model.encoder import axial_resnet_instances
from deeplab2.model.layers import recompute_grad as recompute_grad_lib


def _compute_deeplab_gradients(inputs, model, training):
  """Returns an output and all the gradients."""
  variables = model.trainable_weights[::-1] + [inputs]
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(variables)
    out = model(inputs, training=training)['transformer_mask_feature']

  grads = tape.gradient(out, variables)
  return out, grads


class RecomputeGradTest(tf.test.TestCase):

  def test_real_deeplab_models(self):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
      # Test max_deeplab_s since it involves all three types of operations:
      # convolution, axial-attention, and transformer.
      model_name = 'max_deeplab_s'
      kwargs = {'num_blocks': [1, 1, 1, 1],
                'backbone_layer_multiplier': 1,
                'width_multiplier': 1.0,
                'bn_layer': tf.keras.layers.experimental.SyncBatchNormalization,
                'conv_kernel_weight_decay': 0.0,
                'block_group_config': {
                    'drop_path_keep_prob': 1.0,  # Disable the randomness.
                    'conv_use_recompute_grad': False,
                    'axial_use_recompute_grad': False,
                    'recompute_within_stride': 0,
                    'transformer_use_recompute_grad': False}}
      # Build test input.
      tensor = test_utils.create_test_input(1, 33, 33, 3)
      test_input = tf.Variable(tensor)
      test_input_recompute = tf.Variable(tensor)

      # Build a model.
      model = axial_resnet_instances.get_model(model_name, **kwargs)
      model(test_input, training=True)

      # Set the batch norm gamma as non-zero so that the bottleneck computation
      # affects the output.
      for weight in model.trainable_weights:
        if '/gamma:0' in weight.name:
          weight.assign(tf.ones_like(weight) * 0.1)

      # Activate all recompute_grad for the recomputed model.
      kwargs['block_group_config'] = {
          'drop_path_keep_prob': 1.0,
          'conv_use_recompute_grad': True,
          'axial_use_recompute_grad': True,
          'recompute_within_stride': 0,
          'transformer_use_recompute_grad': True}

      # Build the same model but with recompute_grad.
      model_recompute = axial_resnet_instances.get_model(model_name, **kwargs)
      model_recompute(test_input_recompute, training=True)
      model_recompute.set_weights(model.get_weights())

    @tf.function
    def function():
      outs_recompute, grads_recompute = _compute_deeplab_gradients(
          test_input_recompute, model_recompute, True)
      outs, grads = _compute_deeplab_gradients(
          test_input, model, True)
      return grads_recompute, grads, outs_recompute, outs

    grads_recompute, grads, outs_recompute, outs = strategy.run(function)

    # Similar outputs.
    self.assertAllClose(outs.values[0], outs_recompute.values[0],
                        rtol=1e-4, atol=1e-4)

    # Similar gradients.
    for grad, grad_recompute in zip(grads, grads_recompute):
      if grad is None or grad_recompute is None:
        continue
      grad = grad.values[0]
      grad_recompute = grad_recompute.values[0]
      if (isinstance(grad, tf.IndexedSlices) and
          isinstance(grad_recompute, tf.IndexedSlices)):
        continue
      self.assertAllClose(grad, grad_recompute, rtol=1e-1, atol=1e-1)


def _compute_gradients(model, x):
  with tf.GradientTape() as tape:
    y = model(x)
  return tape.gradient(
      y, model.trainable_variables
      if hasattr(model, 'trainable_variables') else tape.watched_variables())


def _make_gradients_op(model, x):
  f = lambda x: _compute_gradients(model, x)
  return (tf.function(experimental_compile=True)(lambda: f(x))
          if tf.executing_eagerly() else tf.compat.v1.tpu.rewrite(f, (x,)))


class RecomputeDense(tf.keras.layers.Layer):
  """Dense layer that recomputes the forward pass during backpropagation."""

  def __init__(self, units: Sequence[int], **kwargs):
    super(RecomputeDense, self).__init__(**kwargs)
    self._units = tf.nest.flatten(units)

  def build(self, input_shape: tf.TensorShape):
    units = input_shape[-1:] + self._units
    kernels = []
    biases = []
    for i in range(1, len(units)):
      kernels.append(
          self.add_weight('kernel_{}'.format(i), (units[i - 1], units[i])))
      biases.append(self.add_weight('bias_{}'.format(i), (units[i],)))
    self._kernels = kernels
    self._biases = biases
    super(RecomputeDense, self).build(input_shape)

  def call(self, inputs: tf.Tensor, **kwargs):

    @recompute_grad_lib.recompute_grad
    def f(x):
      for kernel, bias in zip(self._kernels, self._biases):
        x = tf.nn.tanh(tf.matmul(x, kernel) + bias)
      return x

    return f(inputs)


class RecomputeDense2Args(RecomputeDense):
  """Extension of `RecomputeDense` that takes and returns 2 arguments."""

  def build(self, input_shape: Sequence[tf.TensorShape]):
    super(RecomputeDense2Args, self).build(input_shape[0])

  def call(self, inputs: Sequence[tf.Tensor], **kwargs):

    @recompute_grad_lib.recompute_grad
    def f(x1, x2):
      for kernel, bias in zip(self._kernels, self._biases):
        x1 = tf.nn.tanh(tf.matmul(x1, kernel) + bias)
      for kernel, bias in zip(self._kernels, self._biases):
        x2 = tf.nn.tanh(tf.matmul(x2, kernel) + bias)
      return x1, x2

    return f(*inputs)


class RecomputeGradXlaTest(tf.test.TestCase):
  """Tests for recompute_grad_lib.recompute_grad with XLA."""

  @property
  def device(self):
    if tf.config.list_logical_devices('TPU'):
      return sorted(tf.config.list_logical_devices('TPU'))[0]
    elif tf.config.list_logical_devices('GPU'):
      return sorted(tf.config.list_logical_devices('GPU'))[0]
    else:
      return sorted(tf.config.list_logical_devices('CPU'))[0]

  def test_xla_model_correctness(self):
    """Tests correctness of the gradient calculation."""

    def _make_model(input_size):
      inputs = tf.keras.Input((input_size,))
      x = inputs
      for _ in range(2):
        x = RecomputeDense([16] * 2)(x)
      outputs = tf.keras.layers.Dense(1)(x)
      return tf.keras.Model(inputs, outputs)

    with tf.device(self.device):
      recompute_model = _make_model(4)
      control_model = tf.keras.Sequential([
          tf.keras.layers.Dense(16, activation='tanh', input_shape=(4,)),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(1),
      ])
      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.tpu.initialize_system())
        self.evaluate(tf.compat.v1.initializers.global_variables())
      for source, target in zip(control_model.trainable_variables,
                                recompute_model.trainable_variables):
        self.evaluate(target.assign(source))
      x = tf.ones((32, 4))
      actual_gradients = self.evaluate(_make_gradients_op(recompute_model, x))
      expected_gradients = self.evaluate(_make_gradients_op(control_model, x))
    for actual, expected in zip(actual_gradients, expected_gradients):
      self.assertAllClose(actual, expected)

  def test_xla_model_2_argument_case(self):
    """Tests for a recomputed function that takes and returns multiple args.

    We don't test correctness of the gradients here; we're just making sure
    `recompute_grad` runs without error in this case.
    """

    def _make_model(input_size):
      input1 = tf.keras.Input((input_size,))
      input2 = tf.keras.Input((input_size,))
      x = (input1, input2)
      for _ in range(2):
        x = RecomputeDense2Args([16] * 2)(x)
      outputs = tf.keras.layers.Dense(1)(x[0] + x[1])
      return tf.keras.Model((input1, input2), outputs)

    with tf.device(self.device):
      recompute_model = _make_model(4)
      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.tpu.initialize_system())
        self.evaluate(tf.compat.v1.initializers.global_variables())
      x1 = tf.ones((32, 4))
      x2 = 2 * tf.ones((32, 4))
      _ = self.evaluate(_make_gradients_op(recompute_model, (x1, x2)))


if __name__ == '__main__':
  tf.test.main()
