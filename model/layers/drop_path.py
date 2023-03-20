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

"""Drop path operation.

This scripts implements the drop path operation, proposed in
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger,
Deep Networks with Stochastic Depth. In ECCV, 2016.
"""
import tensorflow as tf
from deeplab2.model import utils


def get_drop_path_keep_prob(keep_prob_for_last_stage, schedule,
                            current_stage, num_stages):
  """Gets drop path keep probability for current stage.

  Args:
    keep_prob_for_last_stage: A float, the drop path keep probability for
      last stage. This flag is used in conjunction with the flag `schedule`, as
      they together determine drop path keep probability for the other stages.
    schedule: A string, the drop path schedule. Currently, we support
      'constant': use the same drop path keep probability for all stages, and
      'linear': linearly decrease the drop path keep probability from 1.0 at
        0-th stage (or STEM) to `keep_prob_for_last_stage` at last stage.
    current_stage:  An integer, current stage number.
    num_stages: An integer, the number of stages.

  Returns:
    The drop path keep probability for the current stage.

  Raises:
    ValueError: If schedule is not supported.
  """
  if schedule == 'constant':
    return keep_prob_for_last_stage
  elif schedule == 'linear':
    return 1.0 - (1.0 - keep_prob_for_last_stage) * current_stage / num_stages
  else:
    raise ValueError('Unexpected schedule %s.' % schedule)


def generate_drop_path_random_mask(input_tensor, drop_path_keep_prob):
  """Generates a random mask for drop path.

  This function generates a random mask for training models with drop path. Each
  scalar in the output indicates whether the block or path will be kept. The
  scalars are scaled with (1.0 / drop_path_keep_prob) so that the output will
  have the same expectation no mather what the drop_path_keep_prob is.

  Reference:
  "Deep Networks with Stochastic Depth" https://arxiv.org/pdf/1603.09382.pdf

  Args:
    input_tensor: An input [batch_size, n_1, n_2, ..., n_k] tensor.
    drop_path_keep_prob: A float, the keep probability for dropping path.

  Returns:
    binary_tensor: A [batch_size, 1, 1, ..., 1] tensor with the same dtype as
      the input_tensor.
  """
  binary_tensor = None
  if drop_path_keep_prob < 1.0:
    input_shape = input_tensor.get_shape().as_list()
    batch_size = utils.resolve_batch_size(input_tensor)
    random_tensor_shape = [batch_size] + [1] * (len(input_shape) - 1)
    random_tensor = drop_path_keep_prob
    random_tensor += tf.random.uniform(
        random_tensor_shape, dtype=input_tensor.dtype)
    binary_tensor = tf.math.divide(tf.floor(random_tensor), drop_path_keep_prob)
  return binary_tensor


class DropPath(tf.keras.layers.Layer):
  """Drop path layer.

  For details, please see the original paper listed below.
  Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger,
  Deep Networks with Stochastic Depth. In ECCV, 2016.
  """

  def __init__(self, drop_path_keep_prob=1.0, name=None):
    """Initializes a drop path layer.

    Args:
      drop_path_keep_prob: A float, the keep probability for dropping path.
      name: An optional string specifying the operation name.

    Rasies:
      ValueError: If drop_path_keep_prob is <= 0 or > 1.
    """
    super(DropPath, self).__init__(name=name)
    self._drop_path_keep_prob = drop_path_keep_prob
    if self._drop_path_keep_prob <= 0 or self._drop_path_keep_prob > 1.0:
      raise ValueError('drop_path_keep_prob not valid. Got %f.' %
                       self._drop_path_keep_prob)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if training:
      keep_prob = self._drop_path_keep_prob
      shape = (tf.shape(input_tensor)[0],) + (1,) * (
          len(tf.shape(input_tensor)) - 1)
      random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
      random_tensor = tf.floor(random_tensor)
      return (input_tensor / keep_prob) * random_tensor
    return input_tensor

  def get_config(self):
    config = {
        'drop_path_keep_prob': self._drop_path_keep_prob,
    }
    base_config = super(DropPath, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
