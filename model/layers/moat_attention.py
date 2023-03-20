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

"""This file contains the attention used in MOAT [1].

The attention mechanism is based on CoAtNet [2].

[1] MOAT: Alternating Mobile Convolution and Attention
    Brings Strong Vision Models
    arXiv: 2210.01820.
      Chenglin Yang, Siyuan Qiao, Qihang Yu, Xiaoding Yuan,
      Yukun Zhu, Alan Yuille, Hartwig Adam, Liang-Chieh Chen.

[2] CoAtNet: Marrying Convolution and Attention for All Data Sizes
    NeurIPS 2021.
      Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan.
"""

import string
from typing import Optional
import numpy as np
import tensorflow as tf
from deeplab2.utils.hparam_configs import create_config_from_dict

# Global dict storing the computed lookup tensor to avoid repeated computation.
LOOKUP_TENSOR_CACHE = {}


def generate_lookup_tensor(length: int,
                           max_relative_position: Optional[int] = None,
                           dtype: tf.dtypes.DType = tf.float32) -> np.ndarray:
  """Generates one-hot lookup tensor to reindex embeddings along one dimension.

  Args:
    length: The length to reindex to.
    max_relative_position: The maximum relative position to consider.
      Relative position embeddings for distances above this threshold
      are zeroed out.
    dtype: The data type of the returned lookup tensor.

  Returns:
    lookup_tensor: A tensor with shape
    [length, length, relative_position_range]. The element
      satisfies lookup_tensor[n, m, v] = 1{m - n + max_relative_position = v},
      where n, m mean two positions while v the relative position selection.
  """
  if max_relative_position is None:
    max_relative_position = length - 1
  lookup_key = ('lookup_matrix', length, max_relative_position)
  relative_position_range = 2 * max_relative_position + 1
  lookup_tensor_shape = [length, length, relative_position_range]

  if (lookup_key not in LOOKUP_TENSOR_CACHE
      or LOOKUP_TENSOR_CACHE[lookup_key].shape.as_list() != lookup_tensor_shape
      or LOOKUP_TENSOR_CACHE[lookup_key].dtype != dtype):
    lookup_tensor = np.zeros(lookup_tensor_shape)
    for i in range(length):
      for x in range(length):
        v = x - i + max_relative_position
        if abs(x - i) > max_relative_position:
          continue
        lookup_tensor[i, x, v] = 1
    LOOKUP_TENSOR_CACHE[lookup_key] = tf.constant(lookup_tensor, dtype)
  return LOOKUP_TENSOR_CACHE[lookup_key]


def reindex_2d_einsum_lookup(relative_position_tensor: tf.Tensor,
                             height: int,
                             width: int,
                             max_relative_height: Optional[int] = None,
                             max_relative_width: Optional[int] = None,
                             h_axis: int = 0) -> tf.Tensor:
  """Reindexes 2d relative position bias with 2 independent einsum lookups.

  Args:
    relative_position_tensor: A tensor of shape
      [..., relative_position_embedding_height,
      relative_position_embedding_width, ...].
    height: The height to reindex to.
    width: The width to reindex to.
    max_relative_height: Maximum relative height.
      Position embeddings corresponding to vertical distances larger
      than max_relative_height are zeroed out. None to disable.
    max_relative_width: Maximum relative width.
      Position embeddings corresponding to horizontal distances larger
      than max_relative_width are zeroed out. None to disable.
    h_axis: Axis corresponding to relative_position_embedding_height.
      Default to 0.

  Returns:
    reindexed_position_embedding: A Tensor of shape
      [..., height * width, height * width, ...]
  """
  height_lookup = generate_lookup_tensor(
      height, max_relative_position=max_relative_height,
      dtype=relative_position_tensor.dtype)
  width_lookup = generate_lookup_tensor(
      width, max_relative_position=max_relative_width,
      dtype=relative_position_tensor.dtype)

  non_spatial_rank = relative_position_tensor.shape.rank - 2
  non_spatial_expr = ''.join(chr(ord('n') + i) for i in range(non_spatial_rank))
  prefix = non_spatial_expr[:h_axis]
  suffix = non_spatial_expr[h_axis:]

  reindexed_tensor = tf.einsum(
      '{0}hw{1},ixh->{0}ixw{1}'.format(prefix, suffix),
      relative_position_tensor, height_lookup, name='height_lookup')
  reindexed_tensor = tf.einsum(
      '{0}ixw{1},jyw->{0}ijxy{1}'.format(prefix, suffix),
      reindexed_tensor, width_lookup, name='width_lookup')

  ret_shape = relative_position_tensor.shape.as_list()
  ret_shape[h_axis] = height * width
  ret_shape[h_axis + 1] = height * width
  reindexed_tensor = tf.reshape(reindexed_tensor, ret_shape)
  return reindexed_tensor


class TrailDense(tf.keras.layers.Layer):
  """A dense layer that projects features in multiple trailing axes.

  This layer projects features from multiple dimensions to multiple dimensions.
  The trailing axes with size n mean the last n dimensions. This layer avoids
  the extra uses of reshape operations.

  A einsum expression string is generated in this layer, examples:
    - For 4D tensors in conv, a common expression would be 'ABCD,DE->ABCE'.
    - For `q/k/v` head projection in multi-head attention with two output
      trailing dimensions, the expression is 'ABC,CDE->ABDE'
    - For `o` output projection in multi-head attention with
      input_begin_axis = -2, the expression is 'ABCD,CDE->ABE'
  """

  def __init__(self,
               output_trailing_dimensions,
               input_begin_axis=-1,
               use_bias=True,
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='dense'):
    """Initializes TrailDense layer.

    Args:
      output_trailing_dimensions: A list of integers, multiple output
        dimensions in trailing axes. This avoids extra reshape
        operation that splits one single output dimension.
      input_begin_axis: A negative integer, the beginning axes of the input.
        This saves extra reshape operation to merge multiple input dimension.
      use_bias: A boolen, whether to use learnable bias in the layer.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      name: A string, layer name.
    """

    super().__init__(name=name)
    self._output_trailing_dimensions = output_trailing_dimensions
    self._input_begin_axis = input_begin_axis
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shape):
    """Creates variables and einsum expression based on input shape."""

    weight_shape = (input_shape[self._input_begin_axis:] +
                    self._output_trailing_dimensions)
    self.weight = self.add_weight(
        name='weight',
        shape=weight_shape,
        initializer=self._kernel_initializer,
        trainable=True)
    if self._use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=self._output_trailing_dimensions,
          initializer=self._bias_initializer,
          trainable=True)

    # Create einsum expression.
    input_rank = input_shape.rank
    shared_size = self._input_begin_axis % input_rank
    i_only_size = input_rank - shared_size
    o_only_size = len(self._output_trailing_dimensions)

    if input_rank + o_only_size >= len(string.ascii_uppercase):
      raise ValueError('Cannot use einsum as input rank + output rank > 26.')
    einsum_str = string.ascii_uppercase[:input_rank + o_only_size]

    offset = 0
    shared_str = einsum_str[offset:offset+shared_size]
    offset += shared_size
    i_only_str = einsum_str[offset:offset+i_only_size]
    offset += i_only_size
    o_only_str = einsum_str[offset:offset+o_only_size]

    input_str = '{}{}'.format(shared_str, i_only_str)
    output_str = '{}{}'.format(shared_str, o_only_str)
    weight_str = '{}{}'.format(i_only_str, o_only_str)
    self.einsum_expr = '{},{}->{}'.format(input_str, weight_str, output_str)

  def call(self, inputs):
    output = tf.einsum(self.einsum_expr, inputs, self.weight)
    if self._use_bias:
      output += self.bias
    return output


class Attention(tf.keras.layers.Layer):
  """Implementation of Attention.

  This layer performs global self-attention [1] on the input. The input shape
  is [batch_size, height, width, channels] and the output shape is
  [batch_size, height * width, channels].

  If one would like to extend the global self-attention to the local window
  attention [2], they could reshape the input to
  [batch_size * num_window, pixel_num_per_window, channels] followed by
  applying this class.

  [1] Attention Is All You Need.
    NeurIPS 2017.
      Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
      Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

  [2] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
    ICCV 2021.
      Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang,
      Stephen Lin, Baining Guo
  """

  def _retrieve_config(self, config):
    """Retrieves the config of Attention.

    Args:
      config: A dictionary containing the following keys.
        -hidden_size: An integer, output channels.
        -head_size: An integer, the head size of attention.
        -relative_position_embedding_type: A string, type of relative position
          embedding in Attention. Only '2d_multi_head' is supported now.
          If None, no relative position embedding will be used.
        -scale_ratio: A float or a list of floats with length 2, scaling factors
          for the position embedding in height and width dimensions. For
          example, [14/14, 16/14] means the position embedding is created for
          window 14 x 14, but will be interpolated to 14 x 16.
        -kernel_initializer: Initializer for the kernel weights matrix.
        -bias_initializer: Initializer for the bias vector.
        -name: A string, layer name.

    Returns:
      A Config class: hparams_config.Config.
    """

    required_keys = ['hidden_size', 'head_size']
    optional_keys = {
        'relative_position_embedding_type': None,
        'scale_ratio': None,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
        'name': 'attention',
    }
    config = create_config_from_dict(config, required_keys, optional_keys)
    return config

  def __init__(self, **config):
    self._config = self._retrieve_config(config)
    super().__init__(name=self._config.name)
    self._config.num_heads = self._config.hidden_size // self._config.head_size

    self._q_proj = TrailDense(
        output_trailing_dimensions=[self._config.num_heads,
                                    self._config.head_size],
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        name='q')
    self._k_proj = TrailDense(
        output_trailing_dimensions=[self._config.num_heads,
                                    self._config.head_size],
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        name='k')
    self._v_proj = TrailDense(
        output_trailing_dimensions=[self._config.num_heads,
                                    self._config.head_size],
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        name='v')
    self._o_proj = TrailDense(
        output_trailing_dimensions=[self._config.hidden_size],
        input_begin_axis=-2,
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        name='o')

    self._q_scale = self._config.head_size ** -0.5

  def build(self, input_shape):
    if self._config.relative_position_embedding_type == '2d_multi_head':
      if input_shape.rank != 4:
        raise ValueError(
            'The input shape should be [batch_size, height, width, channels]')
      input_shape_list = input_shape.as_list()
      height, width = input_shape_list[-3], input_shape_list[-2]
      if self._config.scale_ratio is not None:
        if (isinstance(self._config.scale_ratio, list) and
            len(self._config.scale_ratio) == 2):
          height_scale, width_scale = self._config.scale_ratio
        elif isinstance(self._config.scale_ratio, float):
          height_scale = self._config.scale_ratio
          width_scale = self._config.scale_ratio
        else:
          raise ValueError(
              'scale ratio should be float or list of floats with length 2')

        relative_position_embedding_height = (
            2 * round(height / height_scale) - 1)
        relative_position_embedding_width = (
            2 * round(width / width_scale) - 1)
      else:
        relative_position_embedding_height = 2 * height - 1
        relative_position_embedding_width = 2 * width - 1
      relative_position_embedding_height_axis = 1
      relative_position_embedding_shape = [
          self._config.num_heads,
          relative_position_embedding_height,
          relative_position_embedding_width]
      self.relative_position_embedding = self.add_weight(
          'relative_position_embedding',
          relative_position_embedding_shape,
          initializer=self._config.kernel_initializer,
          trainable=True)
      if self._config.scale_ratio is not None:
        relative_position_embedding = tf.expand_dims(
            self.relative_position_embedding, axis=-1)
        relative_position_embedding = tf.cast(
            tf.image.resize(relative_position_embedding,
                            [2 * height - 1, 2 * width - 1]),
            self.compute_dtype)
        relative_position_embedding = tf.squeeze(relative_position_embedding,
                                                 axis=-1)
      else:
        relative_position_embedding = tf.cast(self.relative_position_embedding,
                                              self.compute_dtype)

      self.reindexed_position_embedding = reindex_2d_einsum_lookup(
          relative_position_embedding, height, width, height - 1, width - 1,
          h_axis=relative_position_embedding_height_axis)
    elif self._config.relative_position_embedding_type is None:
      self.reindexed_position_embedding = None

  def call(self, query, training):
    _, h, w, channels = query.shape.as_list()
    query = tf.reshape(query, [-1, h * w, channels])

    q_heads = self._q_proj(query)
    k_heads = self._k_proj(query)
    v_heads = self._v_proj(query)
    q_heads *= self._q_scale

    attention_logits = tf.einsum('BSNK, BTNK -> BNST', q_heads, k_heads)

    if self.reindexed_position_embedding is not None:
      attention_logits += self.reindexed_position_embedding

    attention_probs = tf.cast(
        tf.nn.softmax(tf.cast(attention_logits, tf.float32), axis=-1),
        attention_logits.dtype)

    attention_out = tf.einsum('BNST, BTNK -> BSNK', attention_probs, v_heads)
    output = self._o_proj(attention_out)
    return output

