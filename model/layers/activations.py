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

"""Defines a set of useful activation functions."""
import functools
import tensorflow as tf


def gelu(input_tensor, approximate=False):
  """Gaussian Error Linear Unit.

  Reference:
  Gaussian Error Linear Units (GELUs), Dan Hendrycks, Kevin Gimpel, arXiv 2016.

  Args:
    input_tensor: A tensor with an arbitrary shape.
    approximate: A boolean, whether to enable approximation.

  Returns:
    The activated input tensor.
  """
  return tf.keras.activations.gelu(input_tensor, approximate=approximate)


def hard_sigmoid(input_tensor):
  """Hard sigmoid activation function.

  Args:
    input_tensor: A tensor with an arbitrary shape.

  Returns:
    The activated input tensor.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  return tf.nn.relu6(input_tensor + tf.constant(3.)) * 0.16667


def relu6(input_tensor):
  """Relu6 activation function.

  Args:
    input_tensor: A tensor with an arbitrary shape.

  Returns:
    The activated input tensor.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  return tf.nn.relu6(input_tensor)


def swish(input_tensor):
  """Swish or SiLU activation function.

  Args:
    input_tensor: A tensor with an arbitrary shape.

  Returns:
    The activated input tensor.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  return tf.nn.silu(input_tensor)


def hard_swish(input_tensor):
  """Hard Swish function.

  Args:
    input_tensor: A tensor with an arbitrary shape.

  Returns:
    The activated input tensor.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  return input_tensor * tf.nn.relu6(
      input_tensor + tf.constant(3.)) * (1. / 6.)


def identity(input_tensor):
  """Identity function.

  Useful for helping in quantization.

  Args:
    input_tensor: A tensor with an arbitrary shape.

  Returns:
    The activated input tensor.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  return tf.identity(input_tensor)


def get_activation(identifier):
  """Gets activation function via input identifier.

  This function returns the specified customized activation function, if there
  is any. Otherwise, tf.keras.activations.get is called.

  Args:
    identifier: A string, name of the activation function.

  Returns:
    The specified activation function.
  """
  if isinstance(identifier, str):
    name_to_fn = {
        'gelu': functools.partial(gelu, approximate=False),
        'approximated_gelu': functools.partial(gelu, approximate=True),
        'silu': swish,
        'swish': swish,
        'hard_swish': hard_swish,
        'relu6': relu6,
        'hard_sigmoid': hard_sigmoid,
        'identity': identity,
        'none': identity,
    }
    identifier = str(identifier).lower()
    if identifier in name_to_fn:
      return name_to_fn[identifier]
  return tf.keras.activations.get(identifier)
